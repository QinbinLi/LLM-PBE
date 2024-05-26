"""Prompt Leakage"""
from data.prompt_leakage import PromptLeakageSysPrompts
# from models.togetherai import TogetherAIModels
from attacks.PromptLeakage.prompt_leakage import PromptLeakage
from models.hf_models import HFModels
from models.chatgpt import ChatGPT
from models.togetherai import TogetherAIModels
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import torch
import wandb
from transformers import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--num_test', default=10, type=int, help='num of sys prompts to extract.')
parser.add_argument('--data', default='blackfriday', type=str, choices=['blackfriday', 'GPTs', 'blackfriday/Academic', 'blackfriday/Business', 'blackfriday/Creative', 'blackfriday/Game', 'blackfriday/Job-Hunting', 'blackfriday/Marketing', 'blackfriday/Productivity-&-life-style', 'blackfriday/Programming'])
parser.add_argument('--model', default="meta-llama/Llama-2-7b-chat-hf", type=str, choices=[
    # models: https://platform.openai.com/docs/models
    # https://docs.together.ai/docs/inference-models
    # "EleutherAI/pythia-14m",
    # "EleutherAI/pythia-31m",
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-12b",
    "gpt-3.5-turbo",
    "gpt-4",
    # together
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    # not support system prompt
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # 
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-13b-v1.5-16k",
    "togethercomputer/falcon-7b-instruct",
    "togethercomputer/falcon-40b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "claude-2.1",
    ])
parser.add_argument('--defense', default=None, type=str, choices=[
    "no-repeat",
    "top-secret",
    "ignore-ignore-inst",
    "no-ignore",
    "eaten",
])
args = parser.parse_args()

set_seed(args.seed)

args.run_name = args.model.split("/")[-1]+f"_s{args.seed}"
if args.defense is not None:
    args.run_name += f"_d-{args.defense}"
print(f"run name: {args.run_name}")

wandb.init(project='LLM-PBE', name=args.run_name, config=vars(args))

out_dir = f'results/prompt_leakage/{args.data}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

data = PromptLeakageSysPrompts(category=args.data)
sys_prompts = data.random_select(args.num_test, seed=args.seed)

print(f"== model: {args.model} ==")
if 'gpt' in args.model:
    # export OPENAI_API_KEY=XXX
    llm = ChatGPT(model=args.model, max_attempts=30, max_tokens=2048)
# elif 'pythia' in args.model:
    # llm = HFModels(args.model=args.model, max_length=500)
elif 'claude' in args.model:
    from models.claude import ClaudeLLM
    llm = ClaudeLLM(model=args.model)
else:
    # export TOGETHER_API_KEY=XXX
    llm = TogetherAIModels(model=args.model, max_attempts=30, max_tokens=2048)
    # llm = TogetherAIModels(model=args.model, max_attempts=30, max_tokens=256)  # for falcon
attack = PromptLeakage()
# try:
results = attack.execute_attack(sys_prompts, llm)

fname = os.path.join(out_dir, args.run_name  + '.pth')
torch.save(results, fname)

attack_scores = defaultdict(list)
for attack_prompt_name, gen_prompts in results.items():
    match_scores = attack.compute_scores(sys_prompts, gen_prompts)
    attack_scores['attack'].append(attack_prompt_name)
    attack_scores['defense'].append(args.defense)
    attack_scores['match_scores'].append(np.mean(match_scores))
    
    wandb.log({f'{attack_prompt_name}_score': np.mean(match_scores)}, commit=False)

wandb.log({'max_score': np.max(attack_scores['match_scores'])}, commit=True)

df = pd.DataFrame(attack_scores)
print(df)
