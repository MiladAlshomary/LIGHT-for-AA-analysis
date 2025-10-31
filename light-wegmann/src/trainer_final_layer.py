"""
    Based on Original Wegmann Script: https://github.com/nlpsoc/Style-Embeddings/blob/master/src/style_embed/utility/neural_trainer.py
"""

import gc
import logging
import math
import multiprocessing
import os
import psutil
import sys
import time

from random import sample, seed
from typing import NewType, Any

import numpy as np
import pandas as pd
import torch
import transformers
from memory_profiler import profile
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.trainer import TrainerMemoryTracker
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator, TripletEvaluator


from utils.global_const import (
    set_global_seed, 
    SEED, 
    SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, 
    CONVERSATION_U2_COL, CONVERSATION_U1_COL, CONVERSATION_A_COL, 
    ID_U2_COL, ID_U1_COL, ID_A_COL, 
    AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL
)
from utils.global_identifiable import STRANFORMERS_CACHE
from utils.training_const import (
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA, 
    TRIPLET_EVALUATOR, BINARY_EVALUATOR, 
    TRIPLET_LOSS, CONTRASTIVE_ONLINE_LOSS, CONTRASTIVE_LOSS, COSINE_LOSS, 
    UNCASED_TOKENIZER, 
    BATCH_SIZE, EVAL_BATCH_SIZE, EPOCHS, WARMUP_STEPS, 
    LEARNING_RATE, EVALUATION_STEPS, MARGIN, EPS, ROBERTA_BASE
)
from utils.convokit_generator import ANCHOR_COL, U1_COL, U2_COL, SAME_AUTHOR_AU1_COL


class EarlyStopException(Exception):
    pass


# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
# class SentenceBertFineTuner:
class SentenceBertFineTunerStandard:
    """
        base class for fine-tuning sentence-transformer models with AV tasks
    """

    def __init__(self, train_filename, dev_filename, model_path='distilbert-base-nli-stsb-mean-tokens',
                 cache_folder=STRANFORMERS_CACHE, margin=MARGIN, loss=CONTRASTIVE_LOSS,
                 evaluation_type=BINARY_EVALUATOR, debug=False, seed=SEED):
        """

        :param train_filename:
        :param dev_filename:
        :param model_path: When this is called with a HuggingFace model key which is not at the same time a
            sentence transformer this throws a warning. Since this is initializing the model like we want it to
            all the same we allow for such calls, specifically for 'base-bert-cased', 'base-bert-uncased' and
            'roberta-base'
                For example:
                    #   -> "WARNING : No sentence-transformers model found with name
                    #   /home/USER/.cache/torch/sentence_transformers/bert-base-uncased.
                    #   Creating a new one with MEAN pooling."
                    # --> calls _load_auto_model()
        :param save_dir:
        """
        # see also:
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark_continue_training.py

        logging.info(f"setting seed to {seed}")
        self.seed = seed
        set_global_seed(seed=self.seed, w_torch=True)

        if not debug:
            logging.info(f"Calling init from sentence-transformer which is throwing a warning when you use "
                         f"fine-tuning with a base model")
            self.model = SentenceTransformer(model_path, cache_folder=cache_folder)
            self.model.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

            if model_path == ROBERTA_BASE:
                # Roberta seems to be initialized with max seq length of 514 leading to errors in encoding
                self.model.max_seq_length = 512

        self.train_filename = train_filename
        self.dev_filename = dev_filename
        if self.train_filename == self.dev_filename:
            logging.warning('train and dev file are the same ... you should only use this setting for debugging ...')

        if "variable-random" in self.dev_filename and "variable-random" in self.train_filename:
            topic_proxy = "topic-rand"
        elif "variable-subreddit" in self.dev_filename and "variable-subreddit" in self.train_filename:
            topic_proxy = "topic-sub"
        elif "variable-conversation" in self.dev_filename and "variable-conversation" in self.train_filename:
            topic_proxy = "topic-conv"
        else:
            raise ValueError("topic proxy was not uniquely identifiable from train path {} and dev path {} "
                             .format(self.train_filename, self.dev_filename))

        self.loss = loss
        self.evaluation_type = evaluation_type
        if loss == CONTRASTIVE_LOSS or loss == CONTRASTIVE_ONLINE_LOSS or loss == TRIPLET_LOSS:
            loss_param = "loss-{}-margin-{}".format(loss, margin)
            self.margin = margin
        elif loss in [COSINE_LOSS]:  # , SOFTMAX_LOSS
            loss_param = "loss-{}".format(loss)
        else:
            raise ValueError("Given loss function keyword {} not expected ...".format(loss))

        self.save_dir = cache_folder + "av-models/{}/{}-{}-evaluator-{}/seed-{}".format(topic_proxy, model_path,
                                                                                        loss_param,
                                                                                        self.evaluation_type,
                                                                                        self.seed)

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = EARLY_STOPPING_PATIENCE
        self.early_stop_delta = EARLY_STOPPING_DELTA
        self.early_stop_triggered = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Model running on device: {device.upper()}")
        if device == "cuda":
            logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    def similarity(self, utt1: str, utt2: str) -> float:
        # self.model.eval()
        emb1 = self.model.encode(utt1, show_progress_bar=False)
        emb2 = self.model.encode(utt2, show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim

    #@profile(backend='tracemalloc')
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE, warmup_steps=WARMUP_STEPS, evaluation_steps=EVALUATION_STEPS,
              learning_rate=LEARNING_RATE, eps=EPS, load_best_model=False, profile=False,
              eval_batch_size=EVAL_BATCH_SIZE, debug_dataloader=False):
        # see also:
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark_continue_training.py
        if profile:
            logging.info("turning gpu profiling on ... ")
            self._memory_tracker = TrainerMemoryTracker()
            self._memory_tracker.start()

        # Convert the dataset to a DataLoader ready for training
        print("self.train_filename", self.train_filename)
        train_examples = self.get_input_examples(self.train_filename, is_eval_task=False, loss=self.loss,
                                                 evaluation_type=self.evaluation_type, tokenizer=self.model.tokenizer)
        print("self.dev_filename", self.dev_filename)
        val_examples = self.get_input_examples(self.dev_filename, is_eval_task=True, as_float=False, loss=self.loss,
                                               evaluation_type=self.evaluation_type, tokenizer=self.model.tokenizer)
        print(f"Evaluating dataset: {len(val_examples)} triplets")
        # Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
        #     sequences to the max length.
        #   SentenceBertEncoder.smart_batching_collate is required for this to work.
        train_dataloader = DataLoader(train_examples,
                                      shuffle=True,
                                      batch_size=batch_size, pin_memory=True)  # ,
        # pin_memory=True)  # ,
        # num_workers=4)  # collate_fn=self.model.smart_batching_collate)
        logging.info(f"Train batches per epoch: {len(train_dataloader)}")

        VAL_SUBSAMPLE_SIZE = 100000
        if len(val_examples) > VAL_SUBSAMPLE_SIZE:
            seed(42)
            logging.info(f"Using subset of {VAL_SUBSAMPLE_SIZE} samples for early stopping (from {len(val_examples)}).")
            val_examples = sample(list(val_examples), VAL_SUBSAMPLE_SIZE)

        self.val_loader = DataLoader(
            val_examples,
            batch_size=eval_batch_size,
            collate_fn=self.model.smart_batching_collate,
            num_workers=min(4, multiprocessing.cpu_count()),
            pin_memory=True,
            shuffle=False
        )
        logging.info(f"Val batches: {len(self.val_loader)}")

        if evaluation_steps <= 0:
            evaluation_steps = len(train_dataloader)

        if debug_dataloader:
            train_dataloader.collate_fn = self.model.smart_batching_collate
            return train_dataloader

        logging.info("Setting loss to {} ...".format(self.loss))
        if self.loss == COSINE_LOSS:
            train_loss = losses.CosineSimilarityLoss(model=self.model)
        elif self.loss == CONTRASTIVE_LOSS:
            train_loss = losses.ContrastiveLoss(model=self.model,
                                                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                                                margin=self.margin)
        elif self.loss == CONTRASTIVE_ONLINE_LOSS:
            train_loss = losses.OnlineContrastiveLoss(model=self.model,
                                                      distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE)
        elif self.loss == TRIPLET_LOSS:
            train_loss = losses.TripletLoss(model=self.model, triplet_margin=self.margin,
                                            distance_metric=losses.TripletDistanceMetric.COSINE)
        else:
            raise ValueError("Given loss function keyword {} not expected ...".format(self.loss))
        
        self.train_loss = train_loss

        # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples)  # , name='sts-dev')
        logging.info("Setting evaluator to {} ...".format(self.evaluation_type))
        if self.evaluation_type == BINARY_EVALUATOR:
            evaluator = BinaryClassificationEvaluator.from_input_examples(val_examples, batch_size=eval_batch_size,
                                                                          show_progress_bar=True)
        elif self.evaluation_type == TRIPLET_EVALUATOR:
            evaluator = TripletEvaluator.from_input_examples(val_examples, batch_size=eval_batch_size,
                                                             show_progress_bar=False)
        else:
            raise ValueError("evaluation_type received unexpected value {}".format(self.evaluation_type))

        self.val_examples = val_examples
        self.eval_batch_size = eval_batch_size

        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # throws warning with newer transformers version
        #   see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
        #   might be possible to avoid with using a different dataloader approach
        #   this should NOT change the result but "just" slow down training

        try:
            logging.info("Entering model.fit()")
            start = time.time()
            self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                        evaluator=evaluator,
                        epochs=epochs,
                        evaluation_steps=evaluation_steps,
                        warmup_steps=warmup_steps,
                        output_path=self.save_dir,
                        save_best_model=True,
                        # optimizer_class=transformers.AdamW
                        #    the above is previous huggingface optimizer that has a correct_bias param
                        #    now the optimizer is the torch class torch.optim.AdamW without a correct_bias param
                        #    see https://github.com/UKPLab/sentence-transformers/commit/bcc6e195a2a105d0513b6219a83bae3f95903d85
                        #    as far as I can tell bias correction is always done for torch.optim.AdamW
                        #        see: https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW
                        optimizer_params={
                            'lr': learning_rate,
                            'eps': eps,
                            # 'correct_bias': correct_bias  # not allowed for transformers.optim.AdamW
                        },
                        callback=self.callback_test
                        )
            logging.info(f"Total training wall-clock time: {time.time() - start:.2f}s")
        except EarlyStopException:
            logging.info("Training stopped early due to convergence.")
        if profile:
            metrics = {}
            self._memory_tracker.stop_and_update_metrics(metrics)
            logging.info(metrics)

        del train_dataloader, val_examples  # train_dataset

        if load_best_model:
            self.model = SentenceTransformer(self.save_dir)

        return self.save_dir

    def callback_test(self, score, epoch, step):
        start = time.time()
        logging.info("score {} epoch {} step {}".format(score, epoch, step))
        logging.info("{}".format(psutil.virtual_memory()))

        self.model.eval()
        val_loader = self.val_loader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Val batches: {len(val_loader)}")

        total_val_loss, total_batches = 0.0, 0
        with torch.no_grad():
            for features, labels in tqdm(val_loader):
                # Triplet / contrastive losses ignore labels, but we keep the
                # signature uniform.
                for feature in features:
                    for key in feature:
                        if isinstance(feature[key], torch.Tensor):
                            feature[key] = feature[key].to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                loss_val = self.train_loss(features, labels)
                total_val_loss += loss_val.item()
                total_batches += 1

        avg_val_loss = total_val_loss / total_batches
        logging.info(f"Validation loss at epoch {epoch}, step {step}: {avg_val_loss:.4f}")
        if math.isnan(avg_val_loss):
            logging.error(f"Validation loss became NaN at epoch {epoch}, step {step}")
            raise RuntimeError("Validation loss is NaN.")

        # early‑stopping bookkeeping
        if avg_val_loss + self.early_stop_delta < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        elapsed = time.time() - start
        logging.info(f"Total validation time: {elapsed:.2f}s")
        logging.info(f"Avg per batch: {elapsed / total_batches:.4f}s")

        if self.epochs_no_improve >= self.early_stop_patience:
            logging.info("Early stopping triggered")
            raise EarlyStopException

        # logging.info("memory: ")
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_input_examples(task_filename, is_eval_task=False, as_float=True, loss=CONTRASTIVE_LOSS,
                           evaluation_type=BINARY_EVALUATOR, tokenizer=UNCASED_TOKENIZER):
        """
        Sentence BERT labels high similarity as 1 (see: https://www.sbert.net/examples/training/sts/README.html)
            --> i.e., using sameauthor=1 as a label here
        :param is_eval_task: whether task_filename includes the tasks that the model is evaluated on,
            e.g., for early stopping
        :param task_filename:
        :param as_float: whether label should be saved as float ... (necessary for cosine loss)
        :return:
        """
        train_examples = []
        print("reading csv")
        task_data = pd.read_csv(task_filename, sep='\t',
                                dtype={ANCHOR_COL: str, U1_COL: str, U2_COL: str,
                                       ID_U1_COL: str, ID_U2_COL: str, ID_A_COL: str,
                                       AUTHOR_A_COL: str, AUTHOR_U2_COL: str, AUTHOR_U1_COL: str,
                                       CONVERSATION_U1_COL: str, CONVERSATION_U2_COL: str, CONVERSATION_A_COL: str,
                                       SUBREDDIT_A_COL: str, SUBREDDIT_U1_COL: str, SUBREDDIT_U2_COL: str,
                                       SAME_AUTHOR_AU1_COL: int})
        print("task_filename", task_filename, "for row_id, row in tqdm(task_data.iterrows()):")
        for row_id, row in tqdm(task_data.iterrows()):
            # naive CUT OFF strings to a maximum of 512 word word pieces to reduce batch memory load
            # --- DEBUGGING WRAPPER BEFORE TOKENIZATION ---
            anchor_text = row[ANCHOR_COL]
            if not isinstance(anchor_text, str):
                print(f"[Invalid Type] Row {row_id} — Anchor is not string: {repr(anchor_text)} ({type(anchor_text)})")
                print(row)
                raise ValueError(f"[Invalid Type] Row {row_id} — Anchor is not string: {repr(anchor_text)} ({type(anchor_text)})")

            try:
                a = anchor_text[:sum([len(t) + 1 for t in tokenizer.tokenize(anchor_text)[:520]])]
            except Exception as e:
                print("=" * 100)
                print(f"[Tokenizer Exception] at row {row_id}")
                print(f"ANCHOR_COL content:\n{repr(anchor_text)}")
                print("\nFull row dict:")
                print(row.to_dict())
                print("=" * 100)
                raise

            #a = row[ANCHOR_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[ANCHOR_COL])[:520]])]
            u1 = row[U1_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[U1_COL])[:520]])]
            u2 = row[U2_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[U2_COL])[:520]])]
            if as_float:
                au1_label = float(row[SAME_AUTHOR_AU1_COL])  # 1 if author A U1 same, i.e., U1 == SA
                au2_label = float(1 - row[SAME_AUTHOR_AU1_COL])
            else:
                au1_label = int(row[SAME_AUTHOR_AU1_COL])
                au2_label = int(1 - row[SAME_AUTHOR_AU1_COL])

            if (loss != TRIPLET_LOSS and not is_eval_task) or \
                    (evaluation_type == BINARY_EVALUATOR and is_eval_task):
                # PAIR LOSS on train data or BINARY EVALUATOR on dev data
                # contrastive expects: ['This is a positive pair', 'Where the distance will be minimized'], label=1
                # cosine expects: ['My first sentence', 'My second related sentence'], label = 0.8, i.e.,
                #   higher label if positive
                if row_id < 1:
                    logging.info('Collating binary examples for {}'.format(task_filename))
                train_examples.append(InputExample(texts=[a, u1],
                                                   label=au1_label))  # label 1 if same
                train_examples.append(InputExample(texts=[a, u2],
                                                   label=au2_label))
            else:
                # TRIPLET LOSS on train data or TRIPLET EVALUATOR on dev data
                if row_id < 1:
                    logging.info('Collating triple examples for {}'.format(task_filename))
                sa_da_ordered = [u1, u2]
                if int(row[SAME_AUTHOR_AU1_COL]) == 0:
                    sa_da_ordered = [u2, u1]
                # expects: ['Anchor 1', 'Positive 1', 'Negative 1'],
                #   where distance between 'Anchor 1' and 'Positive 1' is minimized
                train_examples.append(InputExample(texts=[a, *sa_da_ordered]))

        return np.array(train_examples)

    def save(self, file_dir):
        # for loading: call fintuner class with path (not file name)
        # self.model.save_pretrained(file_dir)
        self.model.save(file_dir)
