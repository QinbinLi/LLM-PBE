# The code is for sanity check of MIA on GPT2 trained on ag_news.
# credits: https://github.com/DunnBC22/NLP_Projects/blob/main/Causal%20Language%20Modeling/AG%20News/GPT2%20Version/GPT2%20-%20AG_News_CLM.ipynb
#
# If you see error:
#   TypeError: can only concatenate str (not "int") to str
# Update datasets to v2.14.6

import torch
import argparse
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BertTokenizer, BertForMaskedLM
from utils import get_neighborhood_score

parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=1_000)
args = parser.parse_args()

now = datetime.now()
root = os.path.join(
    'experiments',
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
if not os.path.isdir(root):
    os.makedirs(root)

# Load the dataset
dataset = load_dataset("ag_news")
print('Training data shape:', dataset['train'].shape)
print('Eval data shape:', dataset['test'].shape)

BLOCK_SIZE = 128
MODEL_CKPT = "gpt2"

MODEL_NAME = MODEL_CKPT.split("/")[-1] + "-clm-ag_news"
set_seed = 42

per_device_train_batch_size = 128
# BATCH_SIZE = 1000 
NUM_OF_EPOCHS = 1 # 3

WEIGHT_DECAY = 0.01
STRATEGY = "epoch"

REPORTS_TO = "wandb"
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)

def tokenizer_function(samples):
    return tokenizer(samples["text"], truncation=True)
    return tokenizer(samples["text"], truncation=True, max_length=args.length)      # [NEW] For testing different sequence lengths

tokenized_ds = dataset.map(tokenizer_function, 
                            batched=True,
                            remove_columns=dataset["train"].column_names,)

print(tokenized_ds["train"])
print(tokenized_ds["test"])

def group_texts(samples):
    concatenated_examples = {k: sum(samples[k], []) for k in samples.keys()}
    total_length = len(concatenated_examples[list(samples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # result = samples
    # result["labels"] = result["input_ids"].copy()
    return result

clm_ds = tokenized_ds.map(
    group_texts,
    batched=True,
    num_proc=10,
    batch_size=4,
    # load_from_cache_file=False,
)
# clm_ds = tokenized_ds
print(clm_ds["train"])
print(clm_ds["test"])

model = (
    AutoModelForCausalLM.from_pretrained("DunnBC22/gpt2-Causal_Language_Model-AG_News")
    ).to(DEVICE)
model.is_causal = True          # [NEW] Necessary for util.generate_neighbors to work

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# [NEW] Load reference model to search for token replacements
search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
search_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(DEVICE)
search_embedder = search_model.bert.embeddings

import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm

def eval_model(set_name):
    dl = DataLoader(clm_ds[set_name].select(list(range(args.num_iters))), batch_size=16, collate_fn=data_collator)

    loss = 0
    total = 1
    model.eval()
    losses = []
    nscores = []
    for batch in tqdm(dl):
        # batch = {k: torch.stack(v).to(DEVICE) for k, v in batch.items()}
        # if tokenizer.pad_token_id is not None:
        #     batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss.item() * len(batch['input_ids'])
        total += len(batch['input_ids'])

        # Get the logits
        logits = outputs.logits
        # move labels to correct device to enable model parallelism
        labels = batch['labels'].to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss_per_token = loss_per_token.view(batch['input_ids'].size(0), batch['input_ids'].size(1) - 1)

        # attention_mask = batch['attention_mask'][..., 1:].contiguous()
        # sample_losses = torch.sum(sample_losses * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        sample_losses = loss_per_token.mean(1)

        losses.extend(sample_losses.cpu().numpy())

        # [NEW] Calculate neighbor scores
        decoded_texts = tokenizer.batch_decode(batch['input_ids'])
        for i in range(len(decoded_texts)):
            nscore, _ = get_neighborhood_score(
                decoded_texts[i], 
                batch['labels'][i], 
                tokenizer, 
                model, 
                search_tokenizer, 
                search_model, 
                search_embedder
            )
            nscores.append(nscore)
    return losses, nscores

# Evaluate model on train and test sets, save data to disk
train_losses, train_nscores = eval_model('train')
np.save(os.path.join(root, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(root, 'train_nscores.npy'), np.array(train_nscores))

test_losses, test_nscores = eval_model('test')
np.save(os.path.join(root, 'test_losses.npy'), np.array(test_losses))
np.save(os.path.join(root, 'test_nscores.npy'), np.array(test_nscores))

# [NEW] Load data and evaluate AUC
# train_losses = np.load(os.path.join(root, 'train_losses.npy')).tolist()
# test_losses = np.load(os.path.join(root, 'test_losses.npy')).tolist()

train_losses = np.load(os.path.join(root, 'train_nscores.npy'))
train_losses = (-train_losses).tolist()
test_losses = np.load(os.path.join(root, 'test_nscores.npy'))
test_losses = (-test_losses).tolist()

from sklearn.metrics import roc_auc_score
losses = train_losses + test_losses
members = [1] * len(train_losses) + [0] * len(test_losses)
scores = [-l for l in losses]
mia_auc = roc_auc_score(members, [-l for l in losses])
print(f"MIA AUC: {mia_auc}")

threshold = np.mean(train_losses)
corrects = [int(l < 0.0049) == m for l, m in zip(losses, members)]
print(f"Acc: {np.mean(corrects)}")

from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(members, scores)
target_fpr = 0.01
for i in range(len(fpr)):
    if fpr[i] >= target_fpr:
        break
print(f"FPR: {fpr[i]:.4f} | TPR: {tpr[i]:.4f}")

fpr, tpr, threshold = roc_curve(members, scores)
target_fpr = 0.01
for i in range(len(fpr)-1, -1, -1):
    if fpr[i] <= target_fpr:
        break
print(f"FPR: {fpr[i]:.4f} | TPR: {tpr[i]:.4f}")
