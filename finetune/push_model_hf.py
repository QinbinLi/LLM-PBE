"""Push models to huggingface"""
import os
import sys
import argparse
from transformers import AutoConfig
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='echr-llama2-7b-dp8')
parser.add_argument('--readme_only', action='store_true')
args = parser.parse_args()

api = HfApi()

root = 'results/'
hf_model_name = args.model_name

repo_id = f"LLM-PBE/{hf_model_name}"
print(f"creating repo: {repo_id}")
api.create_repo(repo_id=repo_id, private=True, repo_type='model', exist_ok=True,)

print(f"Uploading readme")
api.upload_file(
    path_or_fileobj='hf_README.md',
    path_in_repo='README.md',
    repo_id=repo_id,
)

if not args.readme_only:
    file_list = ['adapter_config.json', 'adapter_model.bin', 'generation_config.json', 'config.json']
    for f in file_list:
        print(f"Uploading {f}")
        api.upload_file(
            path_or_fileobj=os.path.join(root, args.model_name, f),
            path_in_repo=f,
            repo_id=repo_id,
        )
