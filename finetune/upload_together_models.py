"""Upload models to huggingface.

@pbe-aws-data
# not strict, you can choose other env
conda activate pubmed
pip install huggingface_hub

cd pbe-model-upload/
python upload_together_v2.py --model=llama-2-7B-echr-undefended --chkpt=4epoch
"""
import os
import argparse
from huggingface_hub import HfApi


parser = argparse.ArgumentParser()
# parser.add_argument('--mode', default='model', choices=['main', 'model'])
parser.add_argument('--model', default='llama-2-7B-echr-scrubbed')
# 'llama-2-7B-chat-echr-scrubbed'
parser.add_argument('--chkpt', default=None, type=str, help='example: checkpoint_138')
args = parser.parse_args()

api = HfApi()
src_root = '/data/'

print(f"Uploading models...")
if args.chkpt is None:
    chkpt = 'main'
    model_path = os.path.join(
        src_root,
        args.model,
    )
else:
    chkpt = args.chkpt
    model_path = os.path.join(
        src_root,
        args.model,
        args.chkpt
    )
assert os.path.exists(model_path), f"Not found: {model_path}"

repo_id = f"LLM-PBE/together-{args.model}"
revision = chkpt

print(f"creating repo: {repo_id}:{revision}")
api.create_repo(repo_id=repo_id, private=True, repo_type='model', exist_ok=True,)
api.create_branch(repo_id, repo_type="model", branch=revision, exist_ok=True)

for f in os.listdir(model_path):
    full_path = f'{model_path}/{f}'
    target_path = f

    print(f"uploading {full_path}\n=> {target_path}")
    api.upload_file(
        path_or_fileobj=full_path,  # "./test-model/README.md",
        path_in_repo=target_path,
        repo_id=repo_id,
        revision=revision,
    )
