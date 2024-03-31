---
library_name: peft
license: mit
---
Internal material. Do not distribute.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
data_name = 'echr'
base_model_name = 'llama2-7b'
defense_method = 'dp8'
model_path = f'LLM-PBE/'
base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf', 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
model = PeftModel.from_pretrained(
        base_model,
        f'LLM-PBE/{data_name}-llama2-7b-{defense_method}',
        device_map='auto',
    )
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
input_ids = tokenizer('Hello! I am a LLM-PBE chatbot!', return_tensors='pt').input_ids.cuda()
outputs = model.generate(input_ids, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## Training procedure

We use [pii-leakage]() package to finetune the llama2.
Note you have to keyinterrupt to force to save. Otherwise, the default trainer will not save the adaptor config.
* llama2
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

### Framework versions

- PEFT 0.5.0

- PEFT 0.5.0

