# Models 

This directory hosts classes and utilities to interact with Large Language Models (LLMs). The function to create a LLM is `LoadModel`. The base class for all LLMs is `LLMBase`.

## LoadModel

### Parameters

- `model_type` (str): The type of the model you want to load (e.g., "ChatGPT", "Llama").
- `api_key` (str, optional): The API key for remote models.
- `model_path` (str, optional): The local path to the model file.

### Example

```python
from llm_pbe.models import LoadModel
chatgpt = LoadModel("ChatGPT", api_key="your-api-key", open_source=False)
```

## LLMBase Class

The `LLMBase` class is intended to serve as a base class for all specific LLMs, such as ChatGPT, Llama, etc.

### Parameters

- `api_key` (str, optional): This is needed for accessing API-based remote models. You will have to provide a valid API key for the respective service.

- `model_path` (str, optional): If the model parameters are accessible, you can provide either a local path or URL where the model file can be found.


### Functionalities

The `LLM` class provides the following functionalities:
- `load_model()`: This method initializes and loads the model as `self.model`.

- `load_local_model()`: This method loads a model locally, either from a file path or from a URL. It initializes self.tokenizer and self.model. Used internally by the `load_model()` method.

- `load_remote_model()`: This method is for initializing a remote model using an API key. The implementation details are model-specific and need to be provided. Used internally by the `load_model()` method.

- `query(text)`: This method queries the model with a given text prompt and returns the model's output.

- `query_local_model(text)`: A helper function to query a local model. Used internally by the `query()` method.

- `query_remote_model(text)`: A helper function to query a closed-source model using its API. Used internally by the `query()` method.


## Directory Structure

- `README.md`: This document.
- `LLMBase.py`: Contains the base class for LLMs.
- `xxx.py`: Contains the advanced class for specific LLMs.

## How to Extend

To implement a new defense, create a new Python file in the relevant subdirectory. Your defense class should inherit from `LLMBase` and override the `query` function as per the logic of the specific defense mechanism. You can override/add more functions as needed (e.g., `load_model()`).

## Examples

For usage examples, please refer to the main project README.
