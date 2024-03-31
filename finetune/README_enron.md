Finetuning Llama-2
======

The repo is based on pii-leakage (S&P). The original readme is at [here](README_pii.md).

Try to install w/ peft
```shell
conda create -n pii-leakage-v1 python=3.10 -y
conda activate pii-leakage-v1
mkdir repo
cd repo
# should install torch first.
# conflict:
# peft: torch>=1.13.0
# dp-transformers: torch<1.13.0,>=1.9.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
git clone git@github.com:jyhong836/pii_leakage.git
cd pii_leakage
pip install -e .  # this will change torch to 1.13. Max torch<2.0
pip install peft  # should be compatible w/ torch 1.13
cd ..
git clone git@github.com:jyhong836/dp-transformers.git
# I manually change the requirements of dp-transformer to torch<=1.13.1
cd dp-transformers  # very necessary!!! including callbacks!
pip install -e .
pip install wandb
# make sure this is installed as 1.13 again
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# If you works on H100, install a newer version of torch. Otherwise, the inference will fail and output zero loss.
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; torch.tensor([12]).cuda()"
```


## Finetuning Llama-2-7b

### Base Model

We need to use lora. To use lora. Refer to `/localscratch/jyhong/projects/PrivateLLM/repo/dp-transformers/examples/nlg-reddit/sample-level-dp/fine-tune-dp.py`.
Some key points:
* For gpt2,
```python
model = convert_gpt2_attention_to_lora(
            model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
```
* llama2: Note you have to keyinterrupt to force to save. Otherwise, the default trainer will not save the adaptor config.
```shell

python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-undefended.yml

# Train Size: 220683, Eval Size: 49041
