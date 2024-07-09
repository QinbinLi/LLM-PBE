"""Synthetic email dataset

from data.SynthEmail import SynthEmailDataset
# Load scrubbed data
ds = SynthEmailDataset(data_path="data/synthemail", pseudonymize=True, mode='scrubbed')
# save time by avoiding PII scanning.
ds = SynthEmailDataset(data_path="data/synthemail", pseudonymize=False, mode='undefended')
"""

import os
from datasets import load_dataset


class SynthEmailDataset:
    def __init__(self, data_path="path_to_synthetic_email_data", sample_duplication_rate=1, pseudonymize=False, mode="undefended"):
        """
        Initialize the Enron dataset.
        
        Parameters:
        - data_path (str): The file path to the Enron dataset. Default is a placeholder.
        - mode (str): scrubbed | undefended
        """
        
        self.data_path = data_path
        print("SynthEmailDataset")
        self.raw_datasets = load_dataset(
            data_path,
            name=mode,
            sample_duplication_rate=sample_duplication_rate,
            pseudonymize=pseudonymize)
    
    def train_set(self):
        return self.raw_datasets['train']
    
    def test_set(self):
        return self.raw_datasets['test']
