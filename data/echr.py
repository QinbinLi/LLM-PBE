"""ECHR dataset

from data.echr import EchrDataset
# Load scrubbed data
ds = EchrDataset(data_path="data/echr", pseudonymize=True, mode='scrubbed')
# save time by avoiding PII scanning.
ds = EchrDataset(data_path="data/echr", pseudonymize=False, mode='undefended')
"""

import os
from datasets import load_dataset


class EchrDataset:
    def __init__(self, data_path="path_to_enron_data", sample_duplication_rate=1, pseudonymize=False, mode="undefended"):
        """
        Initialize the Enron dataset.
        
        Parameters:
        - data_path (str): The file path to the Enron dataset. Default is a placeholder.
        - mode (str): scrubbed | undefended
        """
        
        self.data_path = data_path
        print("echr, EchrDataset")
        self.raw_datasets = load_dataset(
            data_path,
            name=mode,
            sample_duplication_rate=sample_duplication_rate,
            pseudonymize=pseudonymize)
    
    def train_set(self):
        return self.raw_datasets['train']
    
    def test_set(self):
        return self.raw_datasets['test']
