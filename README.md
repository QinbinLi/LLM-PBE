# Overview

<p>
    <a href="https://llm-pbe.github.io/document">
            <img alt="Build" src="https://img.shields.io/badge/1.0-document-orange">
    </a>
    <a href="https://arxiv.org/abs/2408.12787">
            <img alt="Build" src="https://img.shields.io/badge/arXiv-2408.12787-green">
    </a>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.12-PyTorch-orange">
    </a>
</p>

**LLM-PBE** is a toolkit to assess the data privacy of LLMs. The code is used for the [LLM-PBE](https://llm-pbe.github.io/home) [![arXiv](https://img.shields.io/badge/arXiv-2408.12787-green)](https://arxiv.org/abs/2408.12787) benchmark, which was selected as the :trophy: [Best Research Paper Nomination in VLDB 2024](https://llm-pbe.github.io/vldb2024_nomination_Qinbin.pdf).

## Getting Started
 

### Setup Environment

```shell
conda create -n llm-pbe python=3.10 -y
conda activate llm-pbe
# If you encounter the issue of 'kernel image' when running torch on GPU, try to install a proper torch with cuda.
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/microsoft/analysing_pii_leakage.git
pip install wandb accelerate
pip install -r requirements.txt
```



### Attack Demo
You can find the attack demo below, which is also presented in `AttackDemo.py`
```python
from data import JailbreakQueries
from models import TogetherAIModels
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries()
llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="xxx")
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)
```

### Evaluate DP model metrics
```python
dp_evaluation = metrics.Evaluate(attack_dp_metrics, ground_truths=dataset.labels)
# Output results
print(f"Attack metrics on regular model: {evaluation}")
print(f"Attack metrics on DP model: {dp_evaluation}")
```

### Finetuning LLMs

Finetuning code is hosted separately with a different environment setup. Please refer to [Private Finetuning for LLMs (LLM-PFT)](https://github.com/jyhong836/llm-dp-finetune).
