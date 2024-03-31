import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import json
import argparse
from attacks.MIA.member_inference import MemberInferenceAttack, MIAMetric
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer


# PII_TYPES = ['CARDINAL', 'DATE', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART', 'EVENT']
PII_TYPES = ['DATE', 'PERSON', 'LOC']
METRICS = ['PPL', 'REFER', 'LIRA', 'LOWER_CASE']

# Sample dataset
# Replace this with your actual dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_sample', default=1000, type=int, help='use -1 to include all samples')
parser.add_argument('--data', default='echr', type=str, choices=['echr', 'enron'])
parser.add_argument('--model', default='LLM-PBE/echr-llama2-7b-chat-undefended', type=str)
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat-hf', type=str)
parser.add_argument('--peft', default='lora', type=str)
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--n_neighbor', default=50, type=int, help='num of neighbors in neighbor attack')
args = parser.parse_args()

if args.data == 'echr':
    from data.echr import EchrDataset
    ds = EchrDataset(data_path="data/echr", pseudonymize=True) # , mode='scrubbed')
elif args.data == 'enron':
    from data.enron import EnronDataset
    ds = EnronDataset(data_path="data/enron", pseudonymize=True) # , mode='scrubbed')
else:
    raise NotImplementedError(f"data: {args.data}")

train_set = ds.train_set()
if args.num_sample > 0 and args.num_sample < len(train_set):
    train_set = train_set.select(range(args.num_sample))
test_set = ds.test_set()
if args.num_sample > 0 and args.num_sample < len(test_set):
    test_set = test_set.select(range(args.num_sample))

# Compute character length of text
tokenizer = AutoTokenizer.from_pretrained(args.arch)
train_lengths = list(map(lambda x: len(tokenizer(x).input_ids), train_set['text']))
test_lengths = list(map(lambda x: len(tokenizer(x).input_ids), test_set['text']))


def get_tpr(filtered_df):
    fpr, tpr, thresholds = roc_curve(filtered_df['membership'], - filtered_df['score'])
    tpr_fpr = 0
    for fpr_, tpr_, thr_ in zip(fpr, tpr, thresholds):
        if fpr_ > 0.001:
            tpr_fpr = tpr_
            break
    return tpr_fpr


def run(auc_df, metric):
    run_name = f"{metric}_{args.num_sample}"
    if args.max_seq_len != 1024:
        run_name += f"_len{args.max_seq_len}"
    if args.data != 'echr':
        run_name += f"_{args.data}"
    if args.n_neighbor != 50:
        run_name += f"_nn{args.n_neighbor}"
    result_dir = os.path.join("./results/", f"{args.model}_{args.peft}")
    cache_file = os.path.join(result_dir, run_name)

    extraction_accuracy = []
    targets = []

    # from data.echr import EchrDataset
    # ds = EchrDataset(data_path="data/echr", pseudonymize=True)  # , mode='scrubbed')

    # Plot distribution of text length in the dataset
    # df = pd.DataFrame(ds.train_set())
    # df['length'] = df['text'].str.len()
    # print(df)
    # plt.hist(df['text'].str.len(), 50, (0, 3_000))
    # plt.xlabel('Text Length')
    # plt.ylabel('Frequency')
    # plt.savefig('echr_length_distribution.png')
    # exit(0)

    results = torch.load(cache_file)['results']
    assert sum(results['membership'][:args.num_sample]) == args.num_sample, 'Train examples must come before test examples'

    data = {
        'score': results['score'],
        'membership': results['membership'],
        'length': train_lengths + test_lengths
    }

    # Get flags for PII type
    pii_columns = {}
    for col in PII_TYPES:
        train_pii = list(map(lambda x: x != 'ListPII(data=[])', train_set[col]))
        test_pii = list(map(lambda x: x != 'ListPII(data=[])', test_set[col]))
        pii_columns[col] = train_pii + test_pii
    data.update(pii_columns)

    # TODO need to reformat data. Below codes are from Rachel for data extraction. We need to modify them to fit our format.
    df = pd.DataFrame(data)

    aucs = []
    tprs = []
    categories = []
    sizes = []
    member_scores = []
    nonmember_scores = []

    # Compute auc for sequence length
    steps = [-1, 50, 100, 200, float('inf')]
    if args.data == 'enron':
        steps = [-1, 150, 350, 750, float('inf')]
    for i in range(len(steps) - 1):
        lo = steps[i]
        hi = steps[i+1]
        filtered_df = df[(df['length'] > lo) & (df['length'] <= hi)]
        if len(filtered_df['membership'].value_counts()) < 2:        # Must have 2 classes in group for AUC
            auc = 0
        else:
            auc = roc_auc_score(filtered_df['membership'], - filtered_df['score'])
        categories.append(f'Length ({lo}, {hi}]')
        aucs.append(auc)
        tprs.append(get_tpr(filtered_df))
        member_scores.append(np.mean(filtered_df[filtered_df['membership'] == 1]['score']))
        nonmember_scores.append(np.mean(filtered_df[filtered_df['membership'] == 0]['score']))
        sizes.append(len(filtered_df['membership']))

    # Compute auc for each PII type
    for col in PII_TYPES:
        filtered_df = df[df[col] == True]
        if len(filtered_df['membership'].value_counts()) < 2:        # Must have 2 classes in group for AUC
            auc = 0
        else:
            auc = roc_auc_score(filtered_df['membership'], - filtered_df['score'])
        categories.append(col)
        aucs.append(auc)
        tprs.append(get_tpr(filtered_df))
        member_scores.append(np.mean(filtered_df[filtered_df['membership'] == 1]['score']))
        nonmember_scores.append(np.mean(filtered_df[filtered_df['membership'] == 0]['score']))
        sizes.append(len(filtered_df['membership']))

    auc_df['Category'] = categories
    auc_df['Size'] = sizes
    if metric == 'PPL':
        auc_df['Member PPL'] = list(map(lambda x: f'{x:.2f}', member_scores))
        auc_df['Nonmember PPL'] = list(map(lambda x: f'{x:.2f}', nonmember_scores))
    auc_df[metric + ' (AUC)'] = list(map(lambda x: f'{x*100:.1f}\\%', aucs))
    auc_df[metric + ' (TPR)'] = list(map(lambda x: f'{x*100:.1f}\\%', tprs))
    
    # auc = roc_auc_score(df['membership'], - df['score'])
    # df_dict['pii_type'].append('ALL')
    # df_dict['auc'].append(auc)
    # df_dict['size'].append(len(df['membership']))
    # df_dict['mem ratio'].append(sum(df['membership']) * 1. / len(df['membership']))

    # Plot bar chart


auc_df = pd.DataFrame()
for m in tqdm(METRICS):
    run(auc_df, m)
print(auc_df
      .rename(lambda x: x.replace('_', '\\_'), axis=1)
      .to_latex(index=False))





# plt.figure(figsize=(10, 6))
# ax = sns.barplot(auc_df, x='pii_type', y='auc', color='skyblue')
# ax.bar_label(ax.containers[0], labels=auc_df['size'])
# plt.title(f'{args.metric} MIA AUC by PII Type')
# plt.xlabel('PII Type')
# plt.xticks(rotation=90)
# plt.ylabel('AUC')
# plt.ylim(0, 1.05)
# plt.tight_layout()
# plt.savefig(f'{args.result_dir}/{args.num_sample}_pii.png')






exit(0)







# NUM_BINS = 5
# MAX_LENGTH = 12_000


# Manually filter out outliers
# df = df[df['length'] <= MAX_LENGTH]

min_length = df['length'].min()
max_length = df['length'].max()
print(min_length, max_length)
bins = np.linspace(min_length - 1, max_length, NUM_BINS + 1)

# Creating length categories
# df['length_category'] = pd.cut(df['length'], bins=[10, 13, 16, 54])
df['length_category'] = pd.cut(df['length'], bins=bins)

# df['length_category'] = df['length_category'].astype(str)

# Calculating accuracy for each category
# TODO compute MIA AUC instead.
def compute_auc(sf):
    # Sklearn AUC classifies score ABOVE the threshold as positive label, but we want BELOW threshold to be positive
    # Thus, pos_label must be 0 (test example), not 1 (train example)
    if len(sf['membership'].value_counts()) < 2:        # Must have 2 classes in group for AUC
        auc = 0
    else:
        auc = roc_auc_score(sf['membership'], sf['score'])
    return pd.Series({'auc': auc})


category_auc = df.groupby('length_category')[['score', 'membership', 'length']].apply(compute_auc).reset_index()

# print(category_auc)
auc = category_auc['auc']

start = 0
end = len(auc) - 1
# for start in range(len(auc)):
#     if auc[start] > 0:
#         break
# for end in reversed(range(len(auc))):
#     if end == start or auc[end] > 0:
#         break

CATEGORY = 'length'

# # Creating a bar chart with five bars
plt.figure(figsize=(10, 6))
plt.bar(category_auc['length_category'].astype(str)[start:end+1], auc[start:end+1], color='skyblue')
plt.xlabel('Text Length')
plt.ylabel('AUC')
plt.ylim(0, 1.05)
plt.title(f'{args.metric} MIA AUC by Text Length')
plt.xticks(rotation=90)
plt.tight_layout()
folder = f'{args.result_dir}/{CATEGORY}'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(f'{folder}/{args.num_sample}_{args.metric}.png')



"""
### peft=none
LLM-PBE/echr-gpt2-small-dp8
LLM-PBE/echr-gpt2-small-scrubbed
LLM-PBE/echr-gpt2-small-undefended
LLM-PBE/together-llama-2-7B-echr-scrubbed:checkpoint_138
LLM-PBE/together-llama-2-7B-echr-scrubbed:checkpoint_414
LLM-PBE/together-llama-2-7B-echr-scrubbed:final_model
LLM-PBE/together-llama-2-7B-echr-undefended
LLM-PBE/together-llama-2-7B-echr-undefended:4epoch
LLM-PBE/together-llama-2-7B-enron-undefended:checkpoint_1804
LLM-PBE/together-llama-2-7B-enron-undefended:checkpoint_4059

### peft=lora
LLM-PBE/echr-llama2-7b-chat-dp8
LLM-PBE/echr-llama2-7b-chat-scrubbed
LLM-PBE/echr-llama2-7b-chat-undefended
LLM-PBE/echr-llama2-7b-dp8
LLM-PBE/echr-llama2-7b-scrubbed
LLM-PBE/echr-llama2-7b-undefended
LLM-PBE/enron-llama2-7b-dp8
LLM-PBE/enron-llama2-7b-chat-dp8



python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=LOSS
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=PPL
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=LOWER_CASE
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=REFER
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=WINDOW
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=LIRA
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=NEIGHBOR
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=1000 --metric=ZLIB

python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=LOSS
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=PPL
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=LOWER_CASE
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=REFER
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=WINDOW
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=LIRA
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=NEIGHBOR
python -m attacks.MIA.group_result --model=LLM-PBE/together-llama-2-7B-echr-undefended --peft=none --num_sample=10000 --metric=ZLIB


"""