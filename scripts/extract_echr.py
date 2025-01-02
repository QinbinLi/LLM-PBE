from attacks.DataExtraction.enron import EnronDataExtraction
import random 
random.seed(0)
from models.togetherai import TogetherAIModels
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import os 
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"    
import tiktoken
encoding= tiktoken.encoding_for_model("gpt-3.5-turbo")

parser = argparse.ArgumentParser()
parser.add_argument('--num_sample', default=-1, type=int, help='use -1 to include all samples')
parser.add_argument('--model', default='togethercomputer/llama-2-7b-chat', type=str)
parser.add_argument('--data_name', default='echr-name', type=str)
parser.add_argument('--max_token', default=200, type=int)
args = parser.parse_args()
model_path=args.model


llm = TogetherAIModels(model=args.model, api_key="xxx", max_tokens=512)


dataset= load_dataset(f"LLM-PBE/{args.data_name}")

if args.num_sample!=-1:
    subsample_indx = random.sample(range(len(dataset['train'])), args.num_sample)
else:
    subsample_indx= range(len(dataset['train']))


model_card= args.model.split('/')[1]
folder= 'generations/echr'
if not os.path.exists(folder):
    os.mkdir(folder)

output_fname=os.path.join( folder, f'{model_card}_{args.data_name}_num{args.num_sample}_max{args.max_token}.jsonl')
result=[]

for i, indx in enumerate(subsample_indx):
    context= dataset['train'][indx]['prompt']
    # prompt = tokenizer.decode(tokenizer(context)['input_ids'][-args.max_token:]) # only use the last args.max_token tokens
    prompt = encoding.decode(encoding.encode(context)[-args.max_token:])  # only use the last args.max_token tokens

    ground_truth = dataset['train'][indx]['target']
    try:
        res= llm.query(prompt)
        output= res['output']["choices"][0]['text']
        result.append({'idx':indx, 'label':ground_truth, 'output':output})
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


#  python extract_echr.py --model togethercomputer/llama-2-7b-chat --max_token 200 --num_sample 10 --data_name echr-name
# python extract_echr.py --model togethercomputer/llama-2-7b-chat --max_token 200 --num_sample 10 --data_name echr-date
# python extract_echr.py --model togethercomputer/llama-2-7b-chat --max_token 200 --num_sample 10 --data_name echr-loc
