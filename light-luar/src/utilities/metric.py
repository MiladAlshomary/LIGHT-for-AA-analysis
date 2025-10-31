# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from src.utilities.file_utils import Utils as utils
import pandas as pd

def compute_metrics(
    queries: torch.cuda.FloatTensor, 
    targets: torch.cuda.FloatTensor, 
    split: str
) -> dict:
    """Computes all the metrics specified through the cmd-line. 

    Args:
        params (argparse.Namespace): Command-line parameters.
        queries (torch.cuda.FloatTensor): Query embeddings.
        targets (torch.cuda.FloatTensor): Target embeddings.
        split (str): "validation" or "test"
    """
    # get query and target authors
    query_authors = torch.stack(
            [a for x in queries for a in x['ground_truth']]).cpu().numpy()
    target_authors = torch.stack(
            [a for x in targets for a in x['ground_truth']]).cpu().numpy()

    # get all query and target author embeddings
    q_list = torch.stack([e for x in queries
                          for e in x['{}_embedding'.format(split)]]).cpu().numpy()
    q_list = q_list.reshape(-1, q_list.shape[-1])

    t_list = torch.stack([e for x in targets
                          for e in x['{}_embedding'.format(split)]]).cpu().numpy()
    t_list = t_list.reshape(-1, t_list.shape[-1])
    metric_scores = {}
    metric_scores.update(ranking(q_list, t_list, query_authors, target_authors))
    
    return metric_scores

def ranking(queries, 
            targets,
            query_authors, 
            target_authors, 
            metric='cosine', 
):
    num_queries = len(query_authors)
    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)

    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=-1)

    for i in range(num_queries):
        try:
            dist = distances[i]
            sorted_indices = np.argsort(dist)
            sorted_target_authors = target_authors[sorted_indices]
            ranks[i] = np.where(sorted_target_authors ==
                                query_authors[i])[0].item()
            reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        except:
            continue
    

    return_dict = {
        'R@8': np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        'R@16': np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        'R@32': np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        'R@64': np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict

def compute_metrics_multiluar(
    queries: torch.cuda.FloatTensor, 
    targets: torch.cuda.FloatTensor, 
    split: str
) -> dict:
    """Computes all the metrics specified through the cmd-line. 

    Args:
        queries (torch.cuda.FloatTensor): Query embeddings of shape (num_batches * 7, batch_size, embedding_dim).
        targets (torch.cuda.FloatTensor): Target embeddings of shape (num_batches * 7, batch_size, embedding_dim).
        split (str): "validation" or "test".
    """
    # Reshape and concatenate query embeddings
    q_list = torch.cat(
        [e['{}_embedding'.format(split)].permute(1, 0, 2) for e in queries], dim=0
    )  # Permute to (batch_size, num_layers, embedding_size) and concatenate
    q_list = q_list.cpu().numpy()  # Convert to numpy

    # Reshape and concatenate target embeddings
    t_list = torch.cat(
        [e['{}_embedding'.format(split)].permute(1, 0, 2) for e in targets], dim=0
    )  # Permute to (batch_size, num_layers, embedding_size) and concatenate
    t_list = t_list.cpu().numpy()  # Convert to numpy

    # Extract query and target authors
    query_authors = torch.cat(
        [a['ground_truth'] for a in queries]
    ).cpu().numpy()  # Concatenate ground truth for queries

    target_authors = torch.cat(
        [a['ground_truth'] for a in targets]
    ).cpu().numpy()  # Concatenate ground truth for targets

    metric_scores = {}
    metric_scores.update(ranking_multiluar_all(q_list, t_list, query_authors, target_authors))
        
    return metric_scores


def ranking_multiluar_avg(queries, 
            targets,
            query_authors, 
            target_authors, 
            metric='cosine',
):
    """
    Perform ranking by comparing each layer's embeddings and combining scores.

    Args:
        queries: Query embeddings of shape (num_queries, num_layers, embedding_dim).
        targets: Target embeddings of shape (num_targets, num_layers, embedding_dim).
        query_authors: Array of query authors.
        target_authors: Array of target authors.
        metric: Metric for distance computation (default: 'cosine').

    Returns:
        dict: A dictionary of ranking metrics.
    """
    num_queries = len(query_authors)

    # Compute the average embeddings across layers
    avg_queries = np.mean(queries, axis=1)  # Shape: (num_queries, embedding_dim)
    avg_targets = np.mean(targets, axis=1)  # Shape: (num_targets, embedding_dim)

    # Compute pairwise distances using the averaged embeddings
    distances = pairwise_distances(avg_queries, Y=avg_targets, metric=metric, n_jobs=-1)

    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)

    for i in range(num_queries):
        try:
            dist = distances[i]  # Use combined distances across layers
            sorted_indices = np.argsort(dist)  # Sort target indices by combined distance
            sorted_target_authors = target_authors[sorted_indices]  # Rank the target authors

            ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item()
            reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        except:
            continue

    return_dict = {
        'R@8': np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        'R@16': np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        'R@32': np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        'R@64': np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict

def ranking_multiluar_all(queries, 
            targets,
            query_authors, 
            target_authors,
            metric='cosine',
):
    """
    Perform ranking by comparing embeddings across all layers and combining cosine similarity scores.

    Args:
        queries: Query embeddings of shape (num_queries, num_layers, embedding_dim).
        targets: Target embeddings of shape (num_targets, num_layers, embedding_dim).
        query_authors: Array of query authors.
        target_authors: Array of target authors.
        metric: Metric for similarity computation (default: 'cosine').

    Returns:
        dict: A dictionary of ranking metrics.
    """
    num_queries, num_layers, _ = queries.shape
    num_targets, _, _ = targets.shape

    # Initialize a similarity matrix to hold the sum of cosine similarities for each query-target pair
    similarity_matrix = np.zeros((num_queries, num_targets), dtype=np.float32)

    for layer in range(num_layers):
        # Compute cosine similarity for the current layer
        layer_similarities = pairwise_distances(queries[:, layer, :], Y=targets[:, layer, :], metric=metric)
        # Add the cosine similarities of this layer to the overall similarity matrix
        similarity_matrix += layer_similarities

    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)

    for i in range(num_queries):
        try:
            # Sort targets by descending similarity (higher similarity is better)
            sorted_indices = np.argsort(similarity_matrix[i])  # Negate to sort in descending order
            sorted_target_authors = target_authors[sorted_indices]  # Rank the target authors

            # Find the rank of the correct target
            ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item()
            reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue

    return_dict = {
        'R@8': np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        'R@16': np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        'R@32': np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        'R@64': np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict

