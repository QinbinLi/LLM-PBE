# import requests
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class HFModels():
    def __init__(self, model_name = None, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parameter = kwargs
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(**inputs, **self.parameter)
        generated_text = self.tokenizer.decode(tokens[0])
        return generated_text