from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time
import os
from copy import deepcopy
import urllib.request

from models.LLMBase import LLMBase

class ClaudeLLM(LLMBase):
    def __init__(self, api_key=None, model = None, max_attempts = 100, max_tokens=256, temperature=0.7):
        super().__init__(api_key=api_key)
        if api_key is not None:
            ClaudeLLM.api_key = api_key
        elif os.environ.get("ANTHROPIC_API_KEY"):
            ClaudeLLM.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            print("No API key provided")
            exit(1)
        self.model = model
        self.payload = {
            "model": model,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
        }
        self.max_attempts = max_attempts
        self.delay_seconds = 3
        
        # wget https://public-json-tokenization-0d8763e8-0d7e-441b-a1e2-1c73b8e79dc3.storage.googleapis.com/claude-v1-tokenization.json
        from transformers import PreTrainedTokenizerFast
        urllib.request.urlretrieve("https://public-json-tokenization-0d8763e8-0d7e-441b-a1e2-1c73b8e79dc3.storage.googleapis.com/claude-v1-tokenization.json", "files/claude-v1-tokenization.json")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="files/claude-v1-tokenization.json")    
    
    def load_model(self):
        pass
        
    def query_remote_model(self, prompt, messages=None):
        if messages is None:
            system_prompt = ""
            user_prompt = prompt
            query_prompt = f"{system_prompt}{HUMAN_PROMPT} {user_prompt}{AI_PROMPT}"
        else:
            query_prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    query_prompt += msg['content']
                elif msg['role'] == 'user':
                    query_prompt += f"{HUMAN_PROMPT} {msg['content']}"
                elif msg['role'] == 'assistant':
                    query_prompt += f"{AI_PROMPT} {msg['content']}"
            query_prompt += f"{AI_PROMPT}"
        
        payload = deepcopy(self.payload)
        payload["prompt"] = query_prompt
        n_attempt = 0
        while n_attempt < self.max_attempts:
            try:
                client = Anthropic(
                    api_key=self.api_key,
                )
                message = client.completions.create(
                    **payload
                )
                response = message.completion
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