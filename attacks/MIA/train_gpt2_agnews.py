# credits: https://github.com/DunnBC22/NLP_Projects/blob/main/Causal%20Language%20Modeling/AG%20News/GPT2%20Version/GPT2%20-%20AG_News_CLM.ipynb

# python train_gpt2_agnews.py
#  @GPU9 https://wandb.ai/jyhong/huggingface/runs/7l9arc11

import math
import torch
import argparse
import numpy as np
from tqdm import trange
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BertTokenizer, BertForMaskedLM
from transformers import TrainerControl, TrainerState, TrainingArguments, Trainer, set_seed, TrainerCallback
from utils import get_neighborhood_score, get_loss
from sklearn.metrics import roc_auc_score
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--test_size', default=-1, type=float, help='ratio of test set. If -1 (default) is provided, use the default split.')
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--mia_type', default='loss', choices=['loss', 'neighbor', 'lira'], type=str)
args = parser.parse_args()

seed = 42
set_seed(seed)

# Load the dataset
dataset = load_dataset("ag_news")
if args.test_size > 0:
    # dataset = concatenate_datasets([dataset['train'], dataset['test']])
    dataset = dataset['train'].train_test_split(args.test_size, seed=seed)
print('Training data shape:', dataset['train'].shape)
print('Eval data shape:', dataset['test'].shape)


# [NEW] Callback to calculate AUC from Neighborhood-MIA attack
def eval(name, num_iters, max_length, causal):
    scores = []
    iterator = iter(dataset.shuffle()[name])
    for _ in trange(num_iters, desc=f'{name} mia'):
        data = next(iterator)
        text = data['text']
        label = data['label']
        with torch.no_grad():
            if args.mia_type == 'loss':
                score = get_loss(text, label, tokenizer, model, max_length, causal=causal)
            elif args.mia_type == 'neighbor':
                score = get_neighborhood_score(text, label, tokenizer, model, search_tokenizer, search_model, search_embedder, max_length, causal=causal)
            elif args.mia_type == 'lira':
                loss = get_loss(text, label, tokenizer, model, max_length, causal=causal)
                ref_loss = get_loss(text, label, tokenizer, ref_model, max_length, causal=causal)
                score = loss - ref_loss
        scores.append(score)
    return scores


class MIACallback(TrainerCallback):
    def __init__(self, num_iters, max_length, eval_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iters = num_iters
        self.max_length = max_length
        self.eval_steps = eval_steps
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        
        train_scores = eval('train', self.num_iters, self.max_length, True)
        test_scores = eval('test', self.num_iters, self.max_length, True)

        ones = np.ones_like(test_scores)
        zeros = np.zeros_like(train_scores)

        y_score = np.concatenate([test_scores, train_scores])
        y_true = np.concatenate([ones, zeros])
        auc = roc_auc_score(y_true, y_score)
        print('AUC:', roc_auc_score(y_true, y_score))
        wandb.log({'mia auc': auc}, commit=False)


MODEL_CKPT = "gpt2"

MODEL_NAME = MODEL_CKPT.split("/")[-1] + "-clm-ag_news"
if args.max_length != 64:
    MODEL_NAME += f"_l{args.max_length}"
if args.test_size > 0:
    MODEL_NAME += f"_ts{args.test_size}"

per_device_train_batch_size = 128

WEIGHT_DECAY = 0.  #  0.01
# STRATEGY = "epoch"
STRATEGY = "steps"

REPORTS_TO = "wandb"

DEVICE = torch.device("cuda")

# [NEW] MIACallback(num_iters, max_length)
mia_callback = MIACallback(1_000, args.max_length, args.eval_steps)

# [NEW] BERT reference model for neighborhood MIA attack
search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
search_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(DEVICE)
search_embedder = search_model.bert.embeddings

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)

def tokenizer_function(samples):
    return tokenizer(samples["text"], truncation=True, max_length=args.max_length)

tokenized_ds = dataset.map(tokenizer_function, 
                            batched=True,
                            remove_columns=dataset["train"].column_names,)

print(tokenized_ds["train"])
print(tokenized_ds["test"])

clm_ds = tokenized_ds

model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT, device_map='auto')
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT, device_map='auto')

targs = TrainingArguments(
    output_dir=MODEL_NAME,
    evaluation_strategy=STRATEGY,
    eval_steps=args.eval_steps,
    learning_rate=args.lr,
    logging_strategy=STRATEGY,
    logging_steps=10,
    save_strategy=STRATEGY,
    save_steps=100,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=args.epochs,
    report_to=REPORTS_TO,
    per_device_train_batch_size=per_device_train_batch_size,
    save_total_limit=1,
)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=clm_ds["train"],
    eval_dataset=clm_ds["test"],
    data_collator=data_collator,
    callbacks=[mia_callback]
)

train_results = trainer.train()

trainer.save_model(MODEL_NAME)

evaluation_results = trainer.evaluate()
print(f"Perplexity: {math.exp(evaluation_results['eval_loss']):.2f}")




# model = (
#     AutoModelForCausalLM.from_pretrained("gpt2-clm-ag_news/", device_map='auto')
#     # AutoModelForCausalLM.from_pretrained("DunnBC22/gpt2-Causal_Language_Model-AG_News")
#     ) # .to(DEVICE)

# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# import numpy as np
# from torch.utils.data import DataLoader

# from tqdm import tqdm

# def eval_model(set_name):
#     dl = DataLoader(clm_ds[set_name], batch_size=64, collate_fn=data_collator)

#     loss = 0
#     total = 1
#     model.eval()
#     losses = []
#     for batch in tqdm(dl):
#         # batch = {k: torch.stack(v).to(DEVICE) for k, v in batch.items()}
#         # if tokenizer.pad_token_id is not None:
#         #     batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
#         batch = {k: v.to(DEVICE) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss += outputs.loss.item() * len(batch['input_ids'])
#         total += len(batch['input_ids'])

#         # Get the logits
#         logits = outputs.logits
#         # move labels to correct device to enable model parallelism
#         labels = batch['labels'].to(logits.device)
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#         loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#         loss_per_token = loss_per_token.view(batch['input_ids'].size(0), batch['input_ids'].size(1) - 1)

#         # attention_mask = batch['attention_mask'][..., 1:].contiguous()
#         # sample_losses = torch.sum(sample_losses * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
#         sample_losses = loss_per_token.mean(1)


#         losses.extend(sample_losses.cpu().numpy())
#     print(loss / total)
#     print(np.mean(losses))
#     return losses

# train_losses = eval_model('train')
# test_losses = eval_model('test')

# from sklearn.metrics import roc_auc_score
# losses = train_losses + test_losses
# members = [1] * len(train_losses) + [0] * len(test_losses)
# scores = [-l for l in losses]
# mia_auc = roc_auc_score(members, [-l for l in losses])
# print(f"MIA AUC: {mia_auc}")

# threshold = np.mean(train_losses)
# corrects = [int(l < 0.0049) == m for l, m in zip(losses, members)]
# print(f"Acc: {np.mean(corrects)}")

# from sklearn.metrics import roc_curve

# fpr, tpr, threshold = roc_curve(members, scores)
# target_fpr = 0.01
# for i in range(len(fpr)):
#     if fpr[i] >= target_fpr:
#         break
# print(f"FPR: {fpr[i]:.4f} | TPR: {tpr[i]:.4f}")

# fpr, tpr, threshold = roc_curve(members, scores)
# target_fpr = 0.01
# for i in range(len(fpr)-1, -1, -1):
#     if fpr[i] <= target_fpr:
#         break
# print(f"FPR: {fpr[i]:.4f} | TPR: {tpr[i]:.4f}")
