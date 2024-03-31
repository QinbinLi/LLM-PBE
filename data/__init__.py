import os
import urllib.request

from .jailbreakqueries import JailbreakQueries

def LoadDataset(dataset_name, dataset_path=None, download_url=None):
    """
    :param dataset_name: Name of the dataset ("MyDataset", "PublicDataset", etc.)
    :param dataset_path: Local path to the dataset. This is where the dataset will be loaded from or downloaded to.
    :param download_url: URL to download the dataset if not found locally.

    :return: Loaded dataset or path to the dataset
    """
    
    if dataset_path is None:
        dataset_path = os.path.join("data", dataset_name + ".txt")  # Use default path with extension

    # Check if the dataset exists locally
    if os.path.exists(dataset_path):
        # Logic to load the dataset from the local file system
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = [line.strip() for line in f.readlines()]
        return dataset  # Return the loaded dataset

    elif download_url is not None:
        # Logic to download the dataset
        print(f"Downloading dataset from {download_url} to {dataset_path}")
        urllib.request.urlretrieve(download_url, dataset_path)
        
        with open(dataset_path, 'r') as f:
            dataset = [line.strip() for line in f.readlines()]
        return dataset  # Return the loaded dataset

    else:
        raise ValueError(f"Dataset {dataset_name} not found locally and no download_url provided.")
