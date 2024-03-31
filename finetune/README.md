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

We finetuned some models from together, mainly by Qinbin. To use this models, first download models (ask Qinbin for API).
Then, unpack [together AI models](https://docs.together.ai/docs/fine-tuning-python#using-a-downloaded-model) for testing.
```shell
cd my-model
zstd -d model.tar.zst
tar -xvf model.tar
cd ..
```


## Finetuning Llama-2-7b

### Base Model

We need to use lora. To use lora. Refer to `/localscratch/jyhong/projects/PrivateLLM/repo/dp-transformers/examples/nlg-reddit/sample-level-dp/fine-tune-dp.py`.
Some key points:
* (Not used) ~~For gpt2,~~
```python
model = convert_gpt2_attention_to_lora(
            model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
```
* llama2: Note you have to keyinterrupt to force to save. Otherwise, the default trainer will not save the adaptor config.
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-undefended.yml
#~ => [09/03] https://wandb.ai/jyhong/huggingface/runs/cox8eqqp
# ~=> https://wandb.ai/jyhong/huggingface/runs/canltrq2
# ~=> https://wandb.ai/jyhong/huggingface/runs/gnw6ovfw
#    @ACES 2xH100 [crashed due to OOM]
# => [09/04] https://wandb.ai/jyhong/huggingface/runs/cvvwis1v
# => https://wandb.ai/jyhong/huggingface/runs/891u4pma
#    @ACES 2xH100
#    PPL: 9.0
# ---- restart ----
# => [09/08] https://wandb.ai/jyhong/huggingface/runs/arzimdzg
#    @ACES 6xA30
#    1200iters eval_loss=1.67, train loss=1.66
# => [09/09] https://wandb.ai/jyhong/huggingface/runs/49ccjkrd 
#    @ACES 2xH100 resume
#    RUNNING: eval_loss=1.65
```
* llama2+dp
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-dp8.yml
# => [09/04] https://wandb.ai/jyhong/huggingface/runs/cxtyk6dq
#    @ACES 1xH100  [24hr?? how many iters?]
#    final train loss: 1.72
# => [09/12] https://wandb.ai/jyhong/huggingface/runs/
#    @ACES-5264 resume and run for 48hours on a single H100.
#    Finished=> dp overflow.
```
* llama2+scrubbing
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-scrubbed.yml
# => [09/09] https://wandb.ai/jyhong/huggingface/runs/wz93y9q1
#    @ACES 2xH100  [24hr?? how many iters?]
# => [09/11] https://wandb.ai/jyhong/huggingface/runs/p1ezuury
# => https://wandb.ai/jyhong/huggingface/runs/lrt92056
#    @ACES-5266 resume and run for 48hours on a single H100.
#    iters:, eval loss=??, final loss=?
```

### Chat Model

* Undefended. Let Qinbing & Chulin test this first.
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-chat-undefended.yml
# => [09/11] https://wandb.ai/jyhong/huggingface/runs/4q2xa06g
#    @ACES-5266 final eval loss: 1.68, train loss: 1.68, Finished, epoch=1.73
# TODO resume and run until epoch=4
# => [09/19] https://wandb.ai/jyhong/huggingface/runs/ubkjfqkp
#    @ACES-5425 resume until 4 epochs.
#    final loss=1.66
```
* dp8
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-chat-dp8.yml
# => [09/11] https://wandb.ai/jyhong/huggingface/runs/nrf2z8u7
#    @ACES-5321 1xH100 final train loss: 1.82, epoch=4
```
* scrubbed
```shell
python fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-chat-scrubbed.yml
# => [09/11] https://wandb.ai/jyhong/huggingface/runs/g4wlvnt2
#    @ACES-5320 1xH100 final eval loss: 1.50, train loss:  1.51, epoch:2.44
# => [09/17] https://wandb.ai/jyhong/huggingface/runs/mqya2pir
#    @ACES-5356 resume and run until epoch=4
```

### Evaluate

* llama2
```shell
python evaluate.py --config_path=configs/evaluate/pii-extraction_llama2_undefended.yml
# => [09/09] https://wandb.ai/jyhong/pii-leakage/runs/vmreh8fz
#    @ACES FIX the loading problem.  100 samples. Only evaluated PPL
# ~~=> [09/09] https://wandb.ai/jyhong/pii-leakage/runs/oyzalose
#    @ACES eval pii Time out
python evaluate.py --config_path=configs/evaluate/pii-extraction_llama2_dp8.yml
# => [09/09] https://wandb.ai/jyhong/pii-leakage/runs/2pu8gllw
#    @ACES FIX the loading problem. Only evaluated PII
# ~~=> [09/09] https://wandb.ai/jyhong/pii-leakage/runs/uoq7wnq8
#    @ACES eval pii Time out
python evaluate.py --config_path=configs/evaluate/pii-extraction_llama2_scrubbed.yml
# => [09/11] https://wandb.ai/jyhong/pii-leakage/runs/h9gusv8h 
#    @ACES-5229

wandb sweep sweeps/evaluate_llama2.yml
# => [09/10] https://wandb.ai/jyhong/pii-leakage/sweeps/o6j1wucu
#    @ACES-5229 eval pii for undefended and dp8
# => [09/16] https://wandb.ai/jyhong/pii-leakage/sweeps/9mn45s51
# => [09/18] https://wandb.ai/jyhong/pii-leakage/sweeps/dtg7f25f
#    @ACES-5321 llama2-chat, {undefended(epoch=1.73), dp8(epoch=4), scrubbed(epoch=4)}
#    scrubbed not finished (due to ACES timeout)
# => [10/03] https://wandb.ai/jyhong/pii-leakage/sweeps/xmcpfmz8
# => https://wandb.ai/jyhong/pii-leakage/sweeps/aqpz5ohe (PPL)
#    @ACES-5825 llama2-chat, {undefended(epoch=4), scrubbed(epoch=4)}
```

### Push model to HF

* base model
```shell
python push_model_hf.py --model_name=echr-llama2-7b-dp8 --readme_only
python push_model_hf.py --model_name=echr-llama2-7b-scrubbed --readme_only
python push_model_hf.py --model_name=echr-llama2-7b-undefended --readme_only
```
* chat model
```shell
python push_model_hf.py --model_name=echr-llama2-7b-chat-dp8 --readme_only
python push_model_hf.py --model_name=echr-llama2-7b-chat-scrubbed --readme_only
python push_model_hf.py --model_name=echr-llama2-7b-chat-undefended --readme_only
```
