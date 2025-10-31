
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, PreTrainedModel

# Adapted LucidRains impl. of Memory Efficient Attention
# https://github.com/lucidrains/memory-efficient-attention-pytorch

from .layers import MemoryEfficientAttention, SelfAttention
from .lightning_trainer import LightningTrainer
from src.utilities.file_utils import Utils as utils

import os

class LUAR(LightningTrainer):
    """Defines the multi-LUAR model.
    """
    
    def __init__(self, params):
        super(LUAR, self).__init__(params)
        self.save_hyperparameters()
        self.create_transformer()
        
        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()
        # self.linear = nn.Linear(self.hidden_size, self.params.embedding_dim)

        self.layer_linear = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.params.embedding_dim) for _ in range(self.transformer.config.num_hidden_layers + 1)]
        )
        
    def create_transformer(self):
        """Creates the Transformer model.
        """
        transformer_modelnames = {
            "roberta": "sentence-transformers/paraphrase-distilroberta-base-v1",
            "roberta_base": "roberta-base",
        }
        modelname = transformer_modelnames[self.params.model_type]

        model_path = os.path.join(utils.transformer_path, modelname)
        self.transformer = AutoModel.from_pretrained(modelname)

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
    
    def get_episode_embeddings(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Computes the Author Embedding. 
        """
        B, E, _ = attention_mask.shape
        input_ids = rearrange(input_ids, 'b e l -> (b e) l')
        attention_mask = rearrange(attention_mask, 'b e l -> (b e) l')

        if document_batch_size > 0:
            outputs = {"last_hidden_state": [], "attentions": [], "hidden_states":[]}
            for i in range(0, len(input_ids), document_batch_size):
                
                out = self.transformer(
                    input_ids=input_ids[i:i+document_batch_size],
                    attention_mask=attention_mask[i:i+document_batch_size],
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
                outputs["last_hidden_state"].append(out["last_hidden_state"])
                outputs["hidden_states"].append(out["hidden_states"])
                if output_attentions:
                    outputs["attentions"].append(out["attentions"])
            
            outputs["last_hidden_state"] = torch.cat(outputs["last_hidden_state"], dim=0)

            outputs["hidden_states"] = tuple([torch.cat([x[i] for x in outputs["hidden_states"]], dim=0) for i in range(len(outputs["hidden_states"][0]))])
            
            if output_attentions:
                outputs["attentions"] = tuple([torch.cat([x[i] for x in outputs["attentions"]], dim=0) for i in range(len(outputs["attentions"][0]))])
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )
        
        all_layer_comment_embeddings = []
        all_layer_episode_embeddings = []

        for layer_idx, layer_hidden_state in enumerate(outputs['hidden_states']):
            # Mean pooling
            layer_comment_embeddings = self.mean_pooling(layer_hidden_state, attention_mask)
            layer_comment_embeddings = rearrange(layer_comment_embeddings, '(b e) l -> b e l', b=B, e=E)
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

        return all_layer_episode_embeddings
    
    def forward(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_episode_embeddings(input_ids, attention_mask, output_attentions, document_batch_size)

        return output

    def _model_forward(self, batch):
        """Passes a batch of data through the model. 
           This is used in the lightning_trainer.py file.
        """
        data, labels = batch

        episode_embeddings, comment_embeddings = self.forward(data)
        labels = torch.flatten(labels)
                
        return episode_embeddings, comment_embeddings