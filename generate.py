"""Demo for generating text from LMs."""
from models.ft_clm import FinetunedCasualLM, PeftCasualLM

# base_path = '/storage/jyhong/projects/projects/PrivateLLM'

# model = FinetunedCasualLM('meta-llama/Llama-2-7b-hf')
# model = FinetunedCasualLM('gpt2')
# model = FinetunedCasualLM(model_path=base_path + '/pii_leakage/results/echr-gpt2-small-undefended', arch='gpt2')
# model = PeftCasualLM(model_path=base_path + '/results/enron_aeslc_llama-2-7b_lora_ppl9_date0829', arch='meta-llama/Llama-2-7b-hf')
model = PeftCasualLM(model_path='LLM-PBE/echr-llama2-7b-dp8', arch='meta-llama/Llama-2-7b-hf')
print(model.query('hello. how are you?'))
