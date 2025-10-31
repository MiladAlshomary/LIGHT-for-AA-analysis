# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from transformers import AutoModel, AutoModelForMaskedLM

from .layers import MemoryEfficientAttention, SelfAttention
from .lightning_trainer import LightningTrainer
from src.utilities.file_utils import Utils as utils

torch._dynamo.config.suppress_errors = True

class Transformer(LightningTrainer):
    """Defines the SBERT model.
    """
    def __init__(self, params):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()
        self.create_transformer()
        
        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()
        self.approach = params.approach
        if self.approach == "multiluar":
            self.layer_linear = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.params.embedding_dim) for _ in range(self.transformer.config.num_hidden_layers+1)]
            )
        else:
            self.linear = nn.Linear(self.hidden_size, self.params.embedding_dim)

    def create_transformer(self):
        """Creates the Transformer model.
        """
        transformer_modelnames = {
            "roberta": "sentence-transformers/paraphrase-distilroberta-base-v1",
            "roberta_base": "roberta-base",
            "modernbert": "answerdotai/ModernBERT-base",
        }
        modelname = transformer_modelnames[self.params.model_type]

        model_path = os.path.join(utils.transformer_path, modelname)

        if self.params.model_type == "modernbert":
            print(modelname)
            self.transformer = AutoModelForMaskedLM.from_pretrained(modelname)
        else:
            self.transformer = AutoModel.from_pretrained(modelname)
        
        self.transformer.to('cuda')

        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads
        
        # if self.params.attention_fn_name != "default":
        #     self.replace_attention()
        
        if self.params.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
    
    def replace_attention(self):
        """Replaces the Transformer's Attention mechanism.
        
           NOTE: This feature has only been tested with the regular 
                 original LUAR SBERT pretrained-model.
        """ 
        attn_fn = {
            "memory_efficient": partial(
                MemoryEfficientAttention, 
                q_bucket_size=32, k_bucket_size=32, 
                heads=self.num_attention_heads, dim_head=self.dim_head,
            ),
        }
        
        if self.params.model_type == "modernbert":
            for i, layer in enumerate(self.transformer.model.layers):
                attention = attn_fn[self.params.attention_fn_name]()
                
                state_dict = layer.attn.state_dict()
                attention.load_state_dict(state_dict, strict=True)

                self.transformer.model.layers[i].attention.self = attention
        else:
            for i, layer in enumerate(self.transformer.encoder.layer):
                attention = attn_fn[self.params.attention_fn_name]()
                
                state_dict = layer.attention.self.state_dict()
                attention.load_state_dict(state_dict, strict=True)

                self.transformer.encoder.layer[i].attention.self = attention
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).float().to(torch.float16)
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_episode_embeddings(self, data):
        """Computes the Author Embedding. """
        # batch_size, num_sample_per_author, episode_length
        input_ids, attention_mask = data[0], data[1]
        B, N, E, _ = input_ids.shape
        
        input_ids = rearrange(input_ids, 'b n e l -> (b n e) l')
        attention_mask = rearrange(attention_mask, 'b n e l -> (b n e) l')

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        if self.approach == "multiluar":
            all_layer_comment_embeddings = []
            all_layer_episode_embeddings = []

            for layer_idx, layer_hidden_state in enumerate(outputs['hidden_states']):
                # Mean pooling
                layer_comment_embeddings = self.mean_pooling(layer_hidden_state, attention_mask)
                layer_comment_embeddings = rearrange(layer_comment_embeddings, '(b n e) l -> (b n) e l', b=B, n=N, e=E)
                all_layer_comment_embeddings.append(layer_comment_embeddings)

                # Attention mechanism and reduce
                layer_episode_embeddings = self.attn_fn(layer_comment_embeddings, layer_comment_embeddings, layer_comment_embeddings)
                layer_episode_embeddings = reduce(layer_episode_embeddings, 'b e l -> b l', 'max')

                # Use the specific linear layer for this transformer layer
                layer_episode_embeddings = self.layer_linear[layer_idx](layer_episode_embeddings)

                # Add a layer dimension
                layer_episode_embeddings = rearrange(layer_episode_embeddings, 'b d -> 1 b d')
                all_layer_episode_embeddings.append(layer_episode_embeddings)

            # Concatenate all layers along the first dimension
            all_layer_episode_embeddings = torch.cat(all_layer_episode_embeddings, dim=0)

            return all_layer_episode_embeddings, all_layer_comment_embeddings
        else:
            # at this point, we're embedding individual "comments"
            comment_embeddings = self.mean_pooling(outputs['hidden_states'][-1], attention_mask)
            comment_embeddings = rearrange(comment_embeddings, '(b n e) l -> (b n) e l', b=B, n=N, e=E)

            # aggregate individual comments embeddings into episode embeddings
            episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
            episode_embeddings = reduce(episode_embeddings, 'b e l -> b l', 'max')

            episode_embeddings = self.linear(episode_embeddings)

            return [episode_embeddings], [comment_embeddings]
        
    # def get_episode_embeddings(self, data):
    #     """Computes embeddings for each layer and returns them."""
    #     input_ids, attention_mask = data[0], data[1]

    #      # Ensure tensors are on the correct device and data type
    #     input_ids = input_ids.to(self.transformer.device).to(torch.long)  # Ensure input_ids is torch.long
    #     attention_mask = attention_mask.to(self.transformer.device).to(torch.float16)

    #     B, N, E, _ = input_ids.shape
        
    #     input_ids = rearrange(input_ids, 'b n e l -> (b n e) l')
    #     attention_mask = rearrange(attention_mask, 'b n e l -> (b n e) l')
        
    #     outputs = self.transformer(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         return_dict=True,
    #         output_hidden_states=True
    #     )
        
    #     all_layer_comment_embeddings = []
    #     all_layer_episode_embeddings = []

    #     for layer_idx, layer_hidden_state in enumerate(outputs['hidden_states']):
    #         # Mean pooling
    #         layer_comment_embeddings = self.mean_pooling(layer_hidden_state, attention_mask)
    #         layer_comment_embeddings = rearrange(layer_comment_embeddings, '(b n e) l -> (b n) e l', b=B, n=N, e=E)
    #         all_layer_comment_embeddings.append(layer_comment_embeddings)

    #         # Attention mechanism and reduce
    #         layer_episode_embeddings = self.attn_fn(layer_comment_embeddings, layer_comment_embeddings, layer_comment_embeddings)
    #         layer_episode_embeddings = reduce(layer_episode_embeddings, 'b e l -> b l', 'max')
            
    #         # Use the specific linear layer for this transformer layer
    #         layer_episode_embeddings = self.layer_linear[layer_idx](layer_episode_embeddings)
            
    #         # Add a layer dimension
    #         layer_episode_embeddings = rearrange(layer_episode_embeddings, 'b d -> 1 b d')
    #         all_layer_episode_embeddings.append(layer_episode_embeddings)

    #     # Concatenate all layers along the first dimension
    #     all_layer_episode_embeddings = torch.cat(all_layer_episode_embeddings, dim=0)

    #     return all_layer_episode_embeddings, all_layer_comment_embeddings

    def forward(self, data):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_episode_embeddings(data)

        return output

    def _model_forward(self, batch):
        """Passes a batch of data through the model. 
           This is used in the lightning_trainer.py file.
        """
        data, labels = batch

        episode_embeddings, comment_embeddings = self.forward(data)
        labels = torch.flatten(labels)
                
        return episode_embeddings, comment_embeddings