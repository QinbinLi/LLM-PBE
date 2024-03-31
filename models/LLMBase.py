import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMBase:
    def __init__(self, api_key=None, model_path=None):
        """
        Initialize a Large Language Model (LLM).
        
        Parameters:
        - api_key (str): The API key for querying closed-source models. Default is None.
        - model_path (str): The file path or URL to the model. Default is None.

        """

        self.api_key = api_key  # API key for accessing LLMs (e.g., ChatGPT)
        self.model_path = model_path  # file path or URL that points to the model
        self.load_model()
    
    def load_model(self):
        if self.model_path:
            self.load_local_model()
        elif self.api_key:
            # self.load_remote_model(self.api_key)
            pass
        elif self.model_name_hf:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_hf)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
        else:
            raise ValueError("Invalid model configuration")

    def load_local_model(self):
        """
        Load model locally from a file path or URL.
        """
        local_path = None  # Initialize to None for safety
        if self.model_path.startswith('http'):
            try:
                # Download the model from the URL and save it locally
                response = requests.get(self.url)
                response.raise_for_status()  # Raise HTTPError for bad responses
                
                # Create a unique filename or check for existing files here if needed
                model_file_name = 'downloaded_model.bin'
                with open(model_file_name, 'wb') as file:
                    file.write(response.content)
                
                local_path = model_file_name  # Update the local path
            except requests.RequestException as e:
                print(f"An error occurred while downloading the model: {e}")
                return
        else:
            local_path = self.model_path
        # Load the tokenizer and model from the file path
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModelForCausalLM.from_pretrained(local_path)

    def load_remote_model(self):
        """Initialize a remote model using an API key."""
        # Implement the code to initialize closed-source model using self.api_key
        pass

    def query(self, text):
        """
        Query a model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        if self.api_key:
            return self.query_remote_model(text)
        else:
            return self.query_local_model(text)
        
    def query_local_model(self, text):
        """
        Query a local model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        
        # Encode the text prompt and generate a response
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        output = self.model.generate(input_ids)
        # Decode the generated text back to a readable string
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def query_remote_model(self, text):
        """
        Query a remote model with a given text prompt using its API.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        
        pass
