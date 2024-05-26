# import requests
import together
import time
from transformers import AutoTokenizer
from copy import deepcopy

from models.LLMBase import LLMBase

def count_tokens(tokenzier, prompt):
    return len(tokenzier.encode(prompt))

class TogetherAIModels(LLMBase):
    def __init__(self, api_key=None, model = None, max_attempts = 100, max_tokens=256, temperature=0.7, top_p=0.7, top_k=50, repetition_penalty=1):
        super().__init__(api_key=api_key)
        if api_key is not None:
            together.api_key = api_key
        self.model = model
        self.payload = {
            "model": model,
            "max_tokens": max_tokens,
            # "stop": "\n\n",
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
        self.max_attempts = max_attempts
        self.delay_seconds = 3
        
        if self.model.startswith('togethercomputer/falcon-'):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.replace('togethercomputer', 'tiiuae'))
        elif 'mistral' in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            # git clone https://github.com/chujiezheng/chat_templates.git
            # to support `system` role.
            chat_template = open('./chat_templates/chat_templates/mistral-instruct.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            except:
                self.tokenizer = None
                print("WARNING: Tokenizer is not founded")
    
    def load_model(self):
        pass
        
    def query_remote_model(self, prompt):
        if self.tokenizer:
            num_tokens = count_tokens(self.tokenizer, prompt)
        else:
            # assume that the default num_tokens is 100 for input prompt
            num_tokens = 100
        payload = deepcopy(self.payload)
        payload["prompt"] = prompt
        n_attempt = 0
        while n_attempt < self.max_attempts:
            try:
                if num_tokens > 4096 - payload['max_tokens']:
                    payload['max_tokens'] = 4096 - num_tokens - 20
                    if payload['max_tokens'] < 1:
                        return ''  # cannot generate anything.
                response = together.Complete.create(**payload)['output']['choices'][0]['text']
            except Exception as e:
                # Catch any exception that might occur and print an error message
                print(f"An error occurred: {e}")
                n_attempt += 1
                time.sleep(self.delay_seconds)
            else:
                break
        if n_attempt == self.max_attempts:
            raise Exception("Max number of attempts reached")
            exit(1)
        return response