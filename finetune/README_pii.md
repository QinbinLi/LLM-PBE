Analyzing Leakage of Personally Identifiable Information in Language Models
====

Source: [github](https://github.com/microsoft/analysing_pii_leakage/tree/main)


## Setup

```shell
conda create -n pii-leakage python=3.10 -y
conda activate pii-leakage
cd ../repo/analysing_pii_leakage
pip install -e .
# If you encounter the issue of 'kernel image' when running torch on GPU, try to install a proper torch with cuda.
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install wandb accelerate
```

To run
```shell
conda activate pii-leakage
cd pii_leakage
```

## Finetuning

We demonstrate how to fine-tune a ```GPT-2 small``` ([Huggingface](https://huggingface.co/gpt2)) model on the [ECHR](https://huggingface.co/datasets/ecthr_cases) dataset
(i) without defenses, (ii) with scrubbing and (iii) with differentially private training (ε=8).

**No Defense**
```shell
python fine_tune.py --config_path configs/fine-tune/echr-gpt2-small-undefended.yml
# => [08/24] https://wandb.ai/jyhong/huggingface/runs/dt6r6oa7
#    @GPU9 [failed due to space]
# => [08/26] https://wandb.ai/jyhong/huggingface/runs/trlv5lc6
# => https://wandb.ai/jyhong/huggingface/runs/fi11y3rn
#    save at /storage/jyhong/projects/projects/PrivateLLM/pii_leakage/results/echr-gpt2-small-undefended
#    final ppl=8.7, loss=2.24
```

**With Scrubbing**

_Note_: All PII will be scrubbed from the dataset. Scrubbing is a one-time operation that requires tagging all PII in the dataset first
which can take many hours depending on your setup. We do not provide tagged datasets.
```shell
python fine_tune.py --config_path configs/fine-tune/echr-gpt2-small-scrubbed.yml
# => https://wandb.ai/jyhong/huggingface/runs/017wrqp2
#    @GPU9
# => [08/26] https://wandb.ai/jyhong/huggingface/runs/80qf1hzl
#    save at /storage/jyhong/projects/projects/PrivateLLM/pii_leakage/results/
#    final ppl=5, loss=1.6 (batch size=16)
# => [08/26] https://wandb.ai/jyhong/huggingface/runs/sp6afrg8
#    save at /storage/jyhong/projects/projects/PrivateLLM/pii_leakage/results/echr-gpt2-small-scrubbed
#    final ppl=5.1, loss=1.75 (batch size=32)
```

**With DP (ε=8.0)**

_Note_: We use the [dp-transformers](https://github.com/microsoft/dp-transformers) wrapper around PyTorch's [opacus](https://github.com/pytorch/opacus) library.
```shell
python fine_tune.py --config_path configs/fine-tune/echr-gpt2-small-dp8.yml
# => [08/28] https://wandb.ai/jyhong/huggingface/runs/za4cmyob
#    @GPU9 
```

## Attacks

Assuming your fine-tuned model is located at ```../echr_undefended``` run the following attacks.
Otherwise, you can edit the ```model_ckpt``` attribute in the ```../configs/<ATTACK>/echr-gpt2-small-undefended.yml``` file to point to the location of the model.

**PII Extraction**

This will extract PII from the model's generated text.
```shell
python extract_pii.py --config_path configs/pii-extraction/echr-gpt2-small-undefended.yml
```

**PII Reconstruction**

This will reconstruct PII from the model given a target sequence.
```shell
$ python reconstruct_pii.py --config_path ../configs/pii-reconstruction/echr-gpt2-small-undefended.yml
```

**PII Inference**

This will infer PII from the model given a target sequence and a set of PII candidates.
```shell
$ python reconstruct_pii.py --config_path ../configs/pii-inference/echr-gpt2-small-undefended.yml
```


## Evaluation

Use the ```evaluate.py``` script to evaluate our privacy attacks against the LM.
```shell
# python evaluate.py --config_path configs/evaluate/pii-extraction.yml

python evaluate.py --config_path configs/evaluate/pii-extraction_undefended.yml
# => [08/28] https://wandb.ai/jyhong/pii-leakage/runs/k8qifpow
#    @GPU9
python evaluate.py --config_path configs/evaluate/pii-extraction_scrubbed.yml
# => [08/28] https://wandb.ai/jyhong/pii-leakage/runs/d6srofz1
#    @GPU9
python evaluate.py --config_path configs/evaluate/pii-extraction_dp8.yml
# => [08/28] 

wandb sweep sweeps/evaluate.yml
# => [09/03] https://wandb.ai/jyhong/pii-leakage/sweeps/d0ympxcu
# => https://wandb.ai/jyhong/pii-leakage/sweeps/ijmd3bi4
#    @GPU9 100 samples
# => [09/03] https://wandb.ai/jyhong/pii-leakage/sweeps/9fcufywu
#    @GPU9 1e4 samples
```
This will compute the precision/recall for PII extraction and accuracy for PII reconstruction/inference attacks.
