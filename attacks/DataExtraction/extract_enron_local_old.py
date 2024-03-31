from attacks.DataExtraction.enron import EnronDataExtraction
import random 
random.seed(0)
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--num_sample', default=-1, type=int, help='use -1 to include all samples')
parser.add_argument('--model', default='./results/llama-2-7B-enron/checkpoint_451', type=str)
parser.add_argument('--min_prompt_len', default=200, type=int)

args = parser.parse_args()
model_path=args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)


model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  trust_remote_code=True, 
  device_map='auto',
) # .to(device)

if "pythia" in args.model:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

enron = EnronDataExtraction(data_path="data/enron")
format=f'prefix-{args.min_prompt_len}'
model_card= args.model.split('/')[-2] + '_' + args.model.split('/')[-1]


prompts, labels = enron.generate_prompts(format=format)
args.num_sample=len(prompts)
output_fname= f'generations/enron/{model_card}_num{args.num_sample}_min{args.min_prompt_len}.jsonl'
result=[]

for i, prompt in enumerate(tqdm(prompts)):
   
    ground_truth = labels[i]
    try:
        # res= llm.query(prompt)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids.to(device), max_length=1024, temperature=0.7).cpu()
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        result.append({'idx':i, 'output':output_text,'label':ground_truth, 'prompt':prompt})
    except Exception as e:
        print(e)
        continue
    
    if i%500==0:
        print(f'Finish {i} samples')
        with open(output_fname, 'w') as outfile:
            for entry in result:
                json.dump(entry, outfile)
                outfile.write('\n')

with open(output_fname, 'w') as outfile:
    for entry in result:
        json.dump(entry, outfile)
        outfile.write('\n')
