from attacks.DataExtraction.enron import EnronDataExtraction
import random 
random.seed(0)
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np 
from models.ft_clm import PeftCasualLM, FinetunedCasualLM
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--num_sample', default=-1, type=int, help='use -1 to include all samples')
parser.add_argument('--model', default='./results/llama-2-7B-enron/checkpoint_451', type=str)
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat-hf', type=str)
parser.add_argument('--peft', default='lora', type=str)
parser.add_argument('--min_prompt_len', default=200, type=int)
parser.add_argument('--max_seq_len', default=1024, type=int)

args = parser.parse_args()

wandb.init(project='LLM-PBE', config=vars(args))

model_path=args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.arch == 'none':
    args.arch = None  # will infer default arch from model.

if args.peft == 'none':
    llm = FinetunedCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
else:
    llm = PeftCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)

enron = EnronDataExtraction(data_path="data/enron")
format=f'prefix-{args.min_prompt_len}'
model_card= args.model.split('/')[-2] + '_' + args.model.split('/')[-1]


prompts, labels = enron.generate_prompts(format=format)
if args.num_sample!=-1 and args.num_sample<len(prompts):
    prompts= prompts[:args.num_sample]
    labels= labels[:args.num_sample]
else:
    args.num_sample=len(prompts)
output_fname= f'generations/enron/{model_card}_num{args.num_sample}_min{args.min_prompt_len}.jsonl'
result=[]

for i, prompt in enumerate(tqdm(prompts)):
   
    ground_truth = labels[i]
    try:
        res= llm.query(prompt)
        result.append({'idx':i, 'output':res,'label':ground_truth, 'prompt':prompt})

    except Exception as e:
        print(e)
        continue
    
    if i%100==0:
        print(f'Finish {i} samples')
        with open(output_fname, 'w') as outfile:
            for entry in result:
                json.dump(entry, outfile)
                outfile.write('\n')

with open(output_fname, 'w') as outfile:
    for entry in result:
        json.dump(entry, outfile)
        outfile.write('\n')

wandb.finish()
