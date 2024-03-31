
# Env

If the PBE env does not work, try this.
```shell
conda create -n llm-pbe python=3.10 -y
conda activate llm-pbe
# If you encounter the issue of 'kernel image' when running torch on GPU, try to install a proper torch with cuda.
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# known conflicts: pii-leakage, dp-transformers, functorch
# dp-transformers 1.0.1 requires torch<1.13.0,>=1.9.1, but you have torch 2.0.0+cu117 which is incompatible.
# functorch 0.2.1 requires torch<1.13,>=1.12.1, but you have torch 2.0.0+cu117 which is incompatible.
# pii-leakage 0.1.0 requires torch<2.0,>=1.10, but you have torch 2.0.0+cu117 which is incompatible.
# pip install git+https://github.com/microsoft/analysing_pii_leakage.git
cd ..
git clone git@github.com:jyhong836/pii_leakage.git
cd pii_leakage
pip install -e .
pip install wandb accelerate peft
```

Sanity-check to see if models work
```shell
# llama2
python -m attacks.MIA.run --metric=PPL --num_sample=30
# Results cache => ./results/LLM-PBE/echr-llama2-7b-chat-undefended_lora/PPL_30
# 100%|| 30/30 [01:15<00:00,  2.50s/it]
# Train avg score: 5.463704133026903
# 100%|| 30/30 [00:46<00:00,  1.55s/it]
# Test avg score: 5.3059123343318095
# results: {'auc': 0.528888888888889}

# gpt2
python -m attacks.MIA.run --model=LLM-PBE/echr-gpt2-small-undefended --arch=gpt2 --metric=LIRA --num_sample=30 --peft=none
# results: {'auc': 0.6055555555555556}

# sweep experiments w/ GPT2
# read this for using sweep: https://github.com/illidanlab/SplitMix#reproduce-results
wandb sweep sweeps/mia_gpt2_sanity.yml
```

Run experiment:

```shell
python -m attacks.MIA.run --metric=PPL --num_sample=30

wandb sweep sweeps/mia.yml
# => [10/24] https://wandb.ai/jyhong/LLM-PBE/sweeps/7jv88uo5
#    @GPU9 3k x2 samples, PPL, LOW_CASE, DP model
# ~=> [10/24] https://wandb.ai/jyhong/LLM-PBE/sweeps/3wpl47r6
#    @GPU9 3k x2 samples, WINDOW
#    [Failed]
# => [10/24] https://wandb.ai/jyhong/LLM-PBE/sweeps/x5qdsjio
#    @GPU9 3k x2 samples, PPL, LOW_CASE, undefended model

# ---- fixed some bugs ----
# => [10/27] https://wandb.ai/jyhong/LLM-PBE/sweeps/8euxh1lx
#    @ACES 3k x2 samples, undefended model
# => [10/28] https://wandb.ai/jyhong/LLM-PBE/sweeps/ecyewm63
#    @ACES all samples, undefended model
# => [10/28] https://wandb.ai/jyhong/LLM-PBE/sweeps/jexmdvy3
#    @ACES all samples, undefended model, fixed the length to 1024
# => [10/31] https://wandb.ai/jyhong/LLM-PBE/sweeps/72r80ct5
#    @GPU9 all samples. all other models

wandb sweep sweeps/mia_gpt2.yml
# => [10/28] https://wandb.ai/jyhong/LLM-PBE/sweeps/jmb5yt32
#    @ACES all samples, all gpt2
# => [10/31] https://wandb.ai/jyhong/LLM-PBE/sweeps/jb0778xu
#    @ACES all samples, gpt2 base model
```

Finetune gpt2
```shell
python train_gpt2_agnews.py
# => [11/06] https://wandb.ai/jyhong/huggingface/runs/m74iw8mt
#    @VITA12
python train_gpt2_agnews.py --max_length=64 --eval_steps=500 --mia_type=loss --test_size=0.5 --lr=1e-4
# => https://wandb.ai/jyhong/huggingface/runs/w0vvegw1
#    ACES 1:30
# => [11/11] https://wandb.ai/jyhong/huggingface/runs/0sv4xtxx 
#    @VITA10 with MIA callbacks
# => [11/12] https://wandb.ai/jyhong/huggingface/runs/0abzlpbt
#    @VITA10 with MIA callbacks, weight_decay=0
# => [11/13] https://wandb.ai/jyhong/huggingface/runs/j1tidiyj
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=loss
# => [11/13] https://wandb.ai/jyhong/huggingface/runs/00axrjvq
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=loss, use 50% data for training
#    best MIA AUC: 60.7%
# => [11/13] https://wandb.ai/jyhong/huggingface/runs/3u4wmxeb 
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=loss, use 50% data for training, use train set only.
python train_gpt2_agnews.py --max_length=64 --eval_steps=500 --mia_type=loss --test_size=0.5 --lr=1e-4
# *=> [11/13] https://wandb.ai/jyhong/huggingface/runs/dsmfxszj
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=loss, use 50% data for training, use train set only. lr=1e-4

python train_gpt2_agnews.py --max_length=64 --eval_steps=500 --mia_type=neighbor --test_size=0.5 --lr=1e-4
# => https://wandb.ai/jyhong/huggingface/runs/bdf2gftw 
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=neighbor, use 50% data for training
# => https://wandb.ai/jyhong/huggingface/runs/vipedopo
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=loss, use 50% data for training, use train set only. lr=1e-4
python train_gpt2_agnews.py --max_length=64 --eval_steps=500 --mia_type=lira --test_size=0.5 --lr=1e-4
# => https://wandb.ai/jyhong/huggingface/runs/zquw9ea2
#    @VITA10 Lira using training dat only
# => https://wandb.ai/jyhong/huggingface/runs/hxfj7dzv
#    @VITA10 with MIA callbacks, weight_decay=0, mia_type=neighbor, use 50% data for training, use train set only. lr=1e-4

# VITA11
conda create -n llm python=3.10 -y 
conda activate llm
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.28.1 datasets evaluate
pip install accelerate wandb
# => loss = 11.4 (step10)

pip install transformers==4.35.0
```

## llama2 echr from together finetuning


1. Unzip model files
```shell
# pip install together
# export TOGETHER_API_KEY=XXX
# python -m attacks.MIA.download_model
cd /data/llama-2-7B-echr-scrubbed/
zstd -d checkpoint_138.tar.zst
mkdir checkpoint_138
mv checkpoint_138.tar checkpoint_138/checkpoint_138.tar
cd checkpoint_138
tar -xvf checkpoint_138.tar
rm checkpoint_138.tar
# rm checkpoint_138.tar.zst
```
2. Upload models
```shell
python upload_models.py --model=llama-2-7B-echr-scrubbed --chkpt=138
# => LLM-PBE/together-llama-2-7B-echr-scrubbed:checkpoint_138
```

Test models
* undefended (model at VITA10)
```shell
python -m attacks.MIA.run --metric=PPL --num_sample=30 --model=checkpoints/llama2-echr --peft=none

wandb sweep sweeps/mia_llama_together.yml
# => [11/14] https://wandb.ai/jyhong/LLM-PBE/sweeps/ve0x5n62
#    @VITA10
# => [11/14] https://wandb.ai/jyhong/LLM-PBE/sweeps/x1i889mt
#    @VITA10 LOWER_CASE
# => https://wandb.ai/jyhong/LLM-PBE/sweeps/semfft4w
#    @VITA10 LIRA
# => https://wandb.ai/jyhong/LLM-PBE/sweeps/qc0kthoj
#    @VITA10 NEIGHBOR

# ====
# undefended 20 epochs
# => [11/26] https://wandb.ai/jyhong/LLM-PBE/sweeps/c4ns7xmj
#    @ACES 10k test
# => [11/27] https://wandb.ai/jyhong/LLM-PBE/sweeps/4mt7zdvf
#    @ACES 1k test
# => [12/01] https://wandb.ai/jyhong/LLM-PBE/sweeps/3wmavgc2
#    @ACES 1k test, neighbor, refer
# undefended 4 epoch
# => [12/3] https://wandb.ai/jyhong/LLM-PBE/sweeps/4ja7614b
#    @ACES-48, 1k test, ppl, lower case, lira, refer
```
* scrubbed (huggingface)
```shell
python -m attacks.MIA.run --metric=PPL --num_sample=30 --model=LLM-PBE/together-llama-2-7B-echr-scrubbed:checkpoint_138 --arch=meta-llama/Llama-2-7b-hf --peft=none

wandb sweep sweeps/mia_llama_together_scrub.yml
# => [11/18] https://wandb.ai/jyhong/LLM-PBE/sweeps/8369m46c
#    @VITA10 scrubbed model at chkpt 138

# TODO test more checkpoints.
# => https://wandb.ai/jyhong/LLM-PBE/sweeps/adcdip5l
#    @ACES 1k test, model 138, 414
# => [12/2] https://wandb.ai/jyhong/LLM-PBE/sweeps/vs1vnp64
#    @ACES 1k, final model, ppl, lower_case, lira, neighbr, refer
# => [12/3] https://wandb.ai/jyhong/LLM-PBE/sweeps/rj6n3wo0
#    @ACES scrub 4epoch
```

## Test llama with lora

```shell
wandb sweep sweeps/mia_llama_chat_lora.yml
# => [11/26] https://wandb.ai/jyhong/LLM-PBE/sweeps/2fa4erec
#    @ACES 10k test
# => [11/27] https://wandb.ai/jyhong/LLM-PBE/sweeps/k44tzcbh
#    @ACES 1k test
# => [11/28] https://wandb.ai/jyhong/LLM-PBE/sweeps/mddggq26
#    @ACES 1k test, Neighbor, refer
wandb sweep sweeps/mia_llama_lora.yml
# => [11/26] https://wandb.ai/jyhong/LLM-PBE/sweeps/3mop82ma
#    @ACES 10k test
# => [11/27] https://wandb.ai/jyhong/LLM-PBE/sweeps/k4z4aog2
#    @ACES 1k test
# => [12/01] https://wandb.ai/jyhong/LLM-PBE/sweeps/dambftnm
#    @ACES 1k test, Neighbor, refer
```

## Enron dataset

### Finetune w/ Lora

setup
```shell
ml load GCCcore/11.3.0 tmux/3.3a
tmux
cd LLM-PBE/finetune
conda activate pii-leakage-v1
export CUDA_VISIBLE_DEVICES=0
```

undefended:
```shell
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-undefended.yml
# => [11/25] https://wandb.ai/jyhong/huggingface/runs/ybzd1nwg
# => [11/30] https://wandb.ai/jyhong/huggingface/runs/roa8cub4
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/dqnv10kf
#    resume from epoch=0.77
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-chat-undefended.yml 
# => [11/30] https://wandb.ai/jyhong/huggingface/runs/480u99s6
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/tosqq7s1
#    resume from epoch=0.77
#    loss: 1.7
```
scrubbed:
```shell
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-scrubbed.yml
# => [11/25] https://wandb.ai/jyhong/huggingface/runs/dg90sns8
# => [11/28] https://wandb.ai/jyhong/huggingface/runs/id9enz4z
# => [11/30] https://wandb.ai/jyhong/huggingface/runs/7o13d7dv => epoch=0.9 
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/3d02u3i0
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-chat-scrubbed.yml
# => [11/25] https://wandb.ai/jyhong/huggingface/runs/zlmgz4vs 
# => [11/28] https://wandb.ai/jyhong/huggingface/runs/kk6duja0
# => [11/30] https://wandb.ai/jyhong/huggingface/runs/sw0y3jg7 => epoch=0.9
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/xsfnsws3
#    loss: 1.48
```
DP
```shell
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-dp8.yml
# => [11/23] https://wandb.ai/jyhong/huggingface/runs/8vspo6k4 
#    @ACES
# => [11/25] https://wandb.ai/jyhong/huggingface/runs/p4dvgqpe 
#    resume
# => [11/28] https://wandb.ai/jyhong/huggingface/runs/gaw0jizq 
#    resume from epoch 1.9
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/qh4otadv
#    resume from epoch 3.99
#    loss: 1.9
python fine_tune.py --config_path=configs/fine-tune/enron-llama2-7b-chat-dp8.yml
# => [11/23] https://wandb.ai/jyhong/huggingface/runs/cc5vde23 
#    @ACES
# => [11/25] https://wandb.ai/jyhong/huggingface/runs/lb3kpmkc 
#    resume
# => [11/28] https://wandb.ai/jyhong/huggingface/runs/3tf6cx1j
#    resume from epoch 1.6
# => [12/2] https://wandb.ai/jyhong/huggingface/runs/zopt03gh
#    resume from epoch 3.99
#    loss: 1.9
```

Upload models to HF
```shell
cd finetune
python push_model_hf.py --model=enron-llama2-7b-chat-dp8
# [12/3] done
python push_model_hf.py --model=enron-llama2-7b-dp8
# [12/3] done
```

### Test with MIA

Full-finetuned models at together:
```shell
wandb sweep sweeps/mia_enron_llama_together.yml
# => [12/2] https://wandb.ai/jyhong/LLM-PBE/sweeps/bt8ohatt
#    @ACES 4 epochs and 10 epoch
# => [12/3] https://wandb.ai/jyhong/LLM-PBE/sweeps/zbt2kpvd
#    @ACES 4 epochs and 10 epoch
```
Lora-finetuned models:
```shell
wandb sweep sweeps/mia_enron_llama_lora.yml
# => [12/3] https://wandb.ai/jyhong/LLM-PBE/sweeps/ddnamujb
#    @ACES-48 DP
wandb sweep sweeps/mia_enron_llama_chat_lora.yml
# => [12/3] https://wandb.ai/jyhong/LLM-PBE/sweeps/tori5cts
#    @ACES-48 DP
```
