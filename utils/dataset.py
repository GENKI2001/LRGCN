"""
Dataset loading utilities for various graph datasets.
"""
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor
import torch_geometric.transforms as T


def load_dataset(dataset_name: str):
    """
    Load a graph dataset based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
    
    Returns:
        tuple: (dataset, canonical_name)
            - dataset: PyTorch Geometric dataset object
            - canonical_name: Canonical name of the dataset
    
    Raises:
        ValueError: If dataset name is not supported
    """
    # Dataset lists
    planetoid_datasets = ["Cora", "Citeseer", "Pubmed"]
    webkb_datasets = ["Texas", "Cornell", "Wisconsin", "Washington"]
    amazon_datasets = ["Photo", "Computers"]
    wikipedia_datasets = ["Chameleon", "Squirrel"]
    actor_datasets = ["Actor"]
    
    # Create mapping dictionaries
    planetoid_map = {name.lower(): name for name in planetoid_datasets}
    webkb_map = {name.lower(): name for name in webkb_datasets}
    amazon_map = {name.lower(): name for name in amazon_datasets}
    wikipedia_map = {name.lower(): name for name in wikipedia_datasets}
    actor_map = {name.lower(): name for name in actor_datasets}
    
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower in planetoid_map:
        canonical_name = planetoid_map[dataset_name_lower]
        dataset = Planetoid(root="./data", name=canonical_name, transform=T.NormalizeFeatures())
    elif dataset_name_lower in webkb_map:
        canonical_name = webkb_map[dataset_name_lower]
        dataset = WebKB(root="./data", name=canonical_name, transform=T.NormalizeFeatures())
    elif dataset_name_lower in amazon_map:
        canonical_name = amazon_map[dataset_name_lower]
        dataset = Amazon(root="./data/Amazon", name=canonical_name, transform=T.NormalizeFeatures())
    elif dataset_name_lower in wikipedia_map:
        canonical_name = wikipedia_map[dataset_name_lower]
        dataset = WikipediaNetwork(root="./data/WikipediaNetwork", name=dataset_name_lower, transform=T.NormalizeFeatures())
    elif dataset_name_lower in actor_map:
        canonical_name = actor_map[dataset_name_lower]
        dataset = Actor(root="./data/Actor", transform=T.NormalizeFeatures())
    else:
        supported_datasets = planetoid_datasets + webkb_datasets + amazon_datasets + wikipedia_datasets + actor_datasets
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported datasets: {supported_datasets}"
        )
    
    return dataset, canonical_name
