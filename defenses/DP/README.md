

Setup environment. Note you need access to customized pii-leakage (jyhong836/pii_leakage.git). Ask Junyuan for access.
```shell
conda create -n pii-leakage-v1 python=3.10 -y
conda activate pii-leakage-v1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install git+git@github.com:jyhong836/pii_leakage.git
mkdir repo
cd repo
git clone git@github.com:jyhong836/pii_leakage.git
cd pii_leakage
# should install torch first.
# conflict:
# peft: torch>=1.13.0
# dp-transformers: torch<1.13.0,>=1.9.1
pip install -e .  # this will change torch to 1.13. Max torch<2.0
pip install peft  # should be compatible w/ torch 1.13
# I manually change the requirements of dp-transformer to torch<=1.13.1
cd repo/dp-transformers
pip install -e .
pip install wandb
# make sure this is installed as 1.13 again
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# use this for H100. Otherwise, the inference will fail and output zero loss.
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; torch.tensor([12]).cuda()"
```

## Finetune llama2-7b w/ lora

For sanity check, you can replace the configure files with gpt2, which runs faster.

llama2: Note you have to keyinterrupt to force to save. Otherwise, the default trainer will not save the adaptor config. If you forgot to save, you can resume from checkpoint and interrupt to save.
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-undefended.yml
```
* llama2+dp
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-dp8.yml
```
* llama2+scrubbing
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-scrubbed.yml
```

