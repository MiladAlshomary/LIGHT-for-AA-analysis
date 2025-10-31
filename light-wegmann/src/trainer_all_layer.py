"""
    All-layer Transformer modification based on Original Wegmann Script: https://github.com/nlpsoc/Style-Embeddings/blob/master/src/style_embed/utility/neural_trainer.py
"""

import gc
import logging
import math
import multiprocessing
import os
import sys
import time
import psutil

from random import sample, seed
from typing import NewType, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from memory_profiler import profile
from torch.utils.data import DataLoader
from transformers.trainer import TrainerMemoryTracker
from tqdm import tqdm


from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator, TripletEvaluator
from sentence_transformers.models import Transformer, Pooling

from utils.training_const import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA
from utils.global_const import (
    set_global_seed, SEED, 
    SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, 
    CONVERSATION_U2_COL, CONVERSATION_U1_COL, CONVERSATION_A_COL, 
    ID_U2_COL, ID_U1_COL, ID_A_COL, 
    AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL
)
from utils.global_identifiable import STRANFORMERS_CACHE
from utils.training_const import (
    TRIPLET_EVALUATOR, BINARY_EVALUATOR, 
    TRIPLET_LOSS, CONTRASTIVE_ONLINE_LOSS, CONTRASTIVE_LOSS, COSINE_LOSS, 
    UNCASED_TOKENIZER, BATCH_SIZE, EVAL_BATCH_SIZE, EPOCHS, 
    WARMUP_STEPS, LEARNING_RATE, EVALUATION_STEPS, MARGIN, EPS, ROBERTA_BASE
)
from utils.convokit_generator import ANCHOR_COL, U1_COL, U2_COL, SAME_AUTHOR_AU1_COL


class EarlyStopException(Exception):
    pass


###############################################################################
# 1) Custom Transformer that forces output_hidden_states=True and captures them
###############################################################################
class TransformerAllLayers(Transformer):
    """
    Subclass of sentence_transformers.models.Transformer that:
      - Sets output_hidden_states=True.
      - Stores the tuple of hidden states in features["all_layer_token_embeddings"].
    """
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = None,
        model_args: dict = None,
        cache_dir: str = None,
        tokenizer_args: dict = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        **kwargs
    ):
        # Remove arguments not supported in older versions of Transformer
        for unsupported_key in ["use_auth_token", "revision", "local_files_only"]:
            if unsupported_key in kwargs:
                kwargs.pop(unsupported_key)

        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            model_args=model_args,
            cache_dir=cache_dir,
            tokenizer_args=tokenizer_args,
            do_lower_case=do_lower_case,
            tokenizer_name_or_path=tokenizer_name_or_path,
            **kwargs
        )

        # Ensure the underlying HF model returns all hidden states
        self.auto_model.config.output_hidden_states = True

    def forward(self, features):
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        token_type_ids = features.get('token_type_ids', None)

        ################################################
        """
        This commented out part was for ModernBERT only approach,
        But following conditional token_type_ids works for both.

        (Despite, this commented out one also works for RoBERTa)
        """
        # Call underlying model
        # model_outputs = self.auto_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     # token_type_ids=token_type_ids
        # )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "modernbert" in self.auto_model.config._name_or_path.lower():
            assert token_type_ids is None, "ModernBERT does not support token_type_ids"
        else:
            # Only add token_type_ids if the model supports it
            if "token_type_ids" in features and "token_type_ids" in self.auto_model.forward.__code__.co_varnames:
                model_inputs["token_type_ids"] = features["token_type_ids"]

        model_outputs = self.auto_model(**model_inputs)
        ################################################

        # Standard final-layer embeddings for pooling
        features['token_embeddings'] = model_outputs.last_hidden_state
        features['cls_token_embeddings'] = features['token_embeddings'][:, 0, :]
        features['attention_mask'] = attention_mask

        # Stores all hidden states
        features['all_layer_token_embeddings'] = model_outputs.hidden_states
        return features


###############################################################################
# 2) Custom SentenceTransformer that uses [TransformerAllLayers, Pooling]
###############################################################################
class SentenceTransformerAllLayers(SentenceTransformer):
    """
    Subclass of SentenceTransformer that constructs a pipeline:
      - TransformerAllLayers
      - Pooling (mean pooling)
    Passing `model_name_or_path=None` to the parent constructor and supply
    the modules list explicitly.
    """
    def __init__(self, model_name_or_path: str, cache_folder: str = None, **kwargs):
        for unsupported_key in ["use_auth_token", "revision", "local_files_only"]:
            if unsupported_key in kwargs:
                kwargs.pop(unsupported_key)

        # 1) Custom Transformer
        transformer_all = TransformerAllLayers(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_folder,
            tokenizer_args={"add_prefix_space": True},
            **kwargs
        )
        # 2) Standard pooling module
        pooling = Pooling(
            word_embedding_dimension=transformer_all.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        # 3) Pass modules explicitly via `modules=[...]`
        super().__init__(
            model_name_or_path=None,    # Important: avoid path check
            modules=[transformer_all, pooling]
        )


##############################
# 3) Multi-layer TripletLoss
##############################
class MultiLayerTripletLoss(losses.TripletLoss):
    """
    Computes TripletLoss over all hidden layers (averaged).
    """
    def forward(self, sentence_features, labels):
        anchor = self.model(sentence_features[0])
        pos    = self.model(sentence_features[1])
        neg    = self.model(sentence_features[2])

        anchor_hidden = anchor["all_layer_token_embeddings"]
        pos_hidden    = pos["all_layer_token_embeddings"]
        neg_hidden    = neg["all_layer_token_embeddings"]

        total_loss = 0.0
        n_layers = len(anchor_hidden)

        for i in range(n_layers):
            anchor_cls = anchor_hidden[i][:, 0, :]
            pos_cls    = pos_hidden[i][:, 0, :]
            neg_cls    = neg_hidden[i][:, 0, :]

            distance_pos = 1 - util.cos_sim(anchor_cls, pos_cls)
            distance_neg = 1 - util.cos_sim(anchor_cls, neg_cls)
            losses_ = torch.relu(distance_pos - distance_neg + self.triplet_margin)
            total_loss += losses_.mean()

        return total_loss / n_layers


#######################
# SentenceBertFineTuner
#######################
class SentenceBertFineTunerAllLayers:
    """
        base class for fine-tuning sentence-transformer models with AV tasks
    """

    def __init__(self, train_filename, dev_filename, model_path='distilbert-base-nli-stsb-mean-tokens',
                 cache_folder=STRANFORMERS_CACHE, margin=MARGIN, loss=CONTRASTIVE_LOSS,
                 evaluation_type=BINARY_EVALUATOR, debug=False, seed=SEED):

        logging.info(f"setting seed to {seed}")
        self.seed = seed
        set_global_seed(seed=self.seed, w_torch=True)

        if not debug:
            logging.info(f"Calling init from sentence-transformer which is throwing a warning when using "
                         f"fine-tuning with a base model")

            # self.model = SentenceTransformer(model_path, cache_folder=cache_folder)
            self.model = SentenceTransformerAllLayers(
                model_name_or_path=model_path,
                cache_folder=cache_folder
            )

            if model_path == ROBERTA_BASE:
                self.model.max_seq_length = 512

        self.train_filename = train_filename
        self.dev_filename = dev_filename
        if self.train_filename == self.dev_filename:
            logging.warning('train and dev file are the same ... only for debugging ...')

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
        elif loss in [COSINE_LOSS]:
            loss_param = "loss-{}".format(loss)
        else:
            raise ValueError("Given loss function keyword {} not expected ...".format(loss))

        self.save_dir = cache_folder + "av-models-all/{}/{}-{}-evaluator-{}/seed-{}".format(
            topic_proxy, model_path, loss_param, self.evaluation_type, self.seed
        )
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
        emb1 = self.model.encode(utt1, show_progress_bar=False)
        emb2 = self.model.encode(utt2, show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim

    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE, warmup_steps=WARMUP_STEPS,
              evaluation_steps=EVALUATION_STEPS, learning_rate=LEARNING_RATE, eps=EPS,
              load_best_model=False, profile=False, eval_batch_size=EVAL_BATCH_SIZE,
              debug_dataloader=False):

        if profile:
            logging.info("turning gpu profiling on ... ")
            self._memory_tracker = TrainerMemoryTracker()
            self._memory_tracker.start()

        print("self.train_filename", self.train_filename)
        train_examples = self.get_input_examples(
            self.train_filename, is_eval_task=False,
            loss=self.loss, evaluation_type=self.evaluation_type,
            tokenizer=self.model.tokenizer
        )
        print("self.dev_filename", self.dev_filename)
        val_examples = self.get_input_examples(
            self.dev_filename, is_eval_task=True,
            as_float=False, loss=self.loss,
            evaluation_type=self.evaluation_type,
            tokenizer=self.model.tokenizer
        )
        print(f"Evaluating dataset: {len(val_examples)} triplets")

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, pin_memory=True)
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
            train_loss = losses.ContrastiveLoss(
                model=self.model,
                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                margin=self.margin
            )
        elif self.loss == CONTRASTIVE_ONLINE_LOSS:
            train_loss = losses.OnlineContrastiveLoss(
                model=self.model,
                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE
            )
        elif self.loss == TRIPLET_LOSS:
            # train_loss = losses.TripletLoss(
            #     model=self.model,
            #     triplet_margin=self.margin,
            #     distance_metric=losses.TripletDistanceMetric.COSINE
            # )
            train_loss = MultiLayerTripletLoss(
                model=self.model,
                triplet_margin=self.margin,
                distance_metric=losses.TripletDistanceMetric.COSINE
            )
        else:
            raise ValueError("Given loss function keyword {} not expected ...".format(self.loss))

        self.train_loss = train_loss

        logging.info("Setting evaluator to {} ...".format(self.evaluation_type))
        if self.evaluation_type == BINARY_EVALUATOR:
            evaluator = BinaryClassificationEvaluator.from_input_examples(
                val_examples, batch_size=eval_batch_size, show_progress_bar=False
            )
        elif self.evaluation_type == TRIPLET_EVALUATOR:
            evaluator = TripletEvaluator.from_input_examples(
                val_examples, batch_size=eval_batch_size, show_progress_bar=False
            )
        else:
            raise ValueError("evaluation_type received unexpected value {}".format(self.evaluation_type))

        self.val_examples = val_examples
        self.eval_batch_size = eval_batch_size

        warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        try:
            logging.info("Entering model.fit()")
            start = time.time()
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=epochs,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=self.save_dir,
                save_best_model=True,
                optimizer_params={
                    'lr': learning_rate,
                    'eps': eps,
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

        del train_dataloader, val_examples

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
    def get_input_examples(task_filename, is_eval_task=False, as_float=True,
                           loss=CONTRASTIVE_LOSS, evaluation_type=BINARY_EVALUATOR,
                           tokenizer=UNCASED_TOKENIZER):
        train_examples = []
        print("reading csv")
        task_data = pd.read_csv(
            task_filename, sep='\t',
            dtype={ANCHOR_COL: str, U1_COL: str, U2_COL: str,
                   ID_U1_COL: str, ID_U2_COL: str, ID_A_COL: str,
                   AUTHOR_A_COL: str, AUTHOR_U2_COL: str, AUTHOR_U1_COL: str,
                   CONVERSATION_U1_COL: str, CONVERSATION_U2_COL: str, CONVERSATION_A_COL: str,
                   SUBREDDIT_A_COL: str, SUBREDDIT_U1_COL: str, SUBREDDIT_U2_COL: str,
                   SAME_AUTHOR_AU1_COL: int}
        )
        print("task_filename", task_filename, "for row_id, row in tqdm(task_data.iterrows()):")
        for row_id, row in tqdm(task_data.iterrows()):
            a = row[ANCHOR_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[ANCHOR_COL])[:520]])]
            u1 = row[U1_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[U1_COL])[:520]])]
            u2 = row[U2_COL][:sum([len(t) + 1 for t in tokenizer.tokenize(row[U2_COL])[:520]])]
            if as_float:
                au1_label = float(row[SAME_AUTHOR_AU1_COL])
                au2_label = float(1 - row[SAME_AUTHOR_AU1_COL])
            else:
                au1_label = int(row[SAME_AUTHOR_AU1_COL])
                au2_label = int(1 - row[SAME_AUTHOR_AU1_COL])

            if (loss != TRIPLET_LOSS and not is_eval_task) or \
               (evaluation_type == BINARY_EVALUATOR and is_eval_task):
                if row_id < 1:
                    logging.info('Collating binary examples for {}'.format(task_filename))
                train_examples.append(InputExample(texts=[a, u1], label=au1_label))
                train_examples.append(InputExample(texts=[a, u2], label=au2_label))
            else:
                if row_id < 1:
                    logging.info('Collating triple examples for {}'.format(task_filename))
                sa_da_ordered = [u1, u2]
                if int(row[SAME_AUTHOR_AU1_COL]) == 0:
                    sa_da_ordered = [u2, u1]
                train_examples.append(InputExample(texts=[a, *sa_da_ordered]))

        return np.array(train_examples)

    def save(self, file_dir):
        self.model.save(file_dir)
