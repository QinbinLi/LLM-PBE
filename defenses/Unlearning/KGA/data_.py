from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding, default_data_collator
import torch

# MAX_DATASET_SIZE = 220000
# TRAIN_SET_SIZE = 200000
# VALID_SET_SIZE = 20000


class TRANS(Dataset):
    def __init__(self, data_file, src, tgt):
        self.data = self.load_data(data_file, src, tgt)

    def load_data(self, data_file, src, tgt):
        Data = {}
        with open(data_file+f".{src}", 'rt', encoding='utf-8') as sf:
            with open(data_file+f".{tgt}", 'rt', encoding='utf-8') as tf:
                for idx, (sline, tline) in enumerate(zip(sf, tf)):
                    sample = {'src': sline.strip(), 'tgt': tline.strip()}
                    Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):

    return DataLoader(
        dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, collate_fn=default_data_collator)


