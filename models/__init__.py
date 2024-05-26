from .chatgpt import ChatGPT
from .togetherai import TogetherAIModels
from .claude import ClaudeLLM

def LoadModel(model_type, model_path=None, api_key=None, custom_model=None):
    """
    :param model_type: Type of model to load ("ChatGPT", "LLaMa", "Custom", etc.)
    :param model_path: Path to the model.
    :param url: URL to the model.
    :param api_key: API Key for remote models.
    :param custom_model: A pre-loaded custom model object (used when model_type is "Custom").

    :return: Instance of the specified model.
    """
    
    if model_type.lower() == "chatgpt":
        from .chatgpt import ChatGPT  # Import the ChatGPT class
        return ChatGPT(api_key=api_key)
        
    elif model_type.lower() == "llama":
        from .lamma import LLaMa  # Import the LLaMa class
        return LLaMa(model_path=model_path)

    elif model_type.lower() == "custom":
        if custom_model is None:
            raise ValueError("For custom models, the 'custom_model' parameter must not be None.")
        return custom_model  # Here, you might want to wrap the custom model into a class that standardizes the interface
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
