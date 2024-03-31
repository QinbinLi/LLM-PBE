[Documentation](https://llm-pbe.github.io/document)

# Overview

**LLM-PBE** is a toolkit to assess the data privacy of LLMs. The components of LLM-PBE are shown below.
![Alt text](docs/images/components.png)

# Getting Started
You can refer to our primary documentation [here](https://llm-pbe.readthedocs.io/en/latest/index.html).
 

## Env

```shell
conda create -n llm-pbe python=3.10 -y
conda activate llm-pbe
# If you encounter the issue of 'kernel image' when running torch on GPU, try to install a proper torch with cuda.
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/microsoft/analysing_pii_leakage.git
pip install wandb accelerate
pip install -r requirements.txt
```




## Attack Demo
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

# Evaluate DP model metrics
dp_evaluation = metrics.Evaluate(attack_dp_metrics, ground_truths=dataset.labels)

# Output results
print(f"Attack metrics on regular model: {evaluation}")
print(f"Attack metrics on DP model: {dp_evaluation}")
``````
