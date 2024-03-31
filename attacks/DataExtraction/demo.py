from attacks.DataExtraction.enron import EnronDataExtraction
from models.togetherai import TogetherAIModels
from attacks.DataExtraction.prompt_extract import PromptExtraction


enron = EnronDataExtraction(data_path="data/enron")

for format in ['prefix-50','0-shot-known-domain-b','0-shot-unknown-domain-c', '3-shot-known-domain-c', '5-shot-unknown-domain-b']:
    prompts, _ = enron.generate_prompts(format=format)
    # Replace api_key with your own API key
    llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="xxx")
    attack = PromptExtraction()
    results = attack.execute_attack(prompts, llm)
    print("results:", results)

