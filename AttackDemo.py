from attacks.DataExtraction.enron import EnronDataExtraction
from models.togetherai import TogetherAIModels
from attacks.Jailbreak.jailbreak import Jailbreak

enron = EnronDataExtraction(data_path="data/enron")
prompts, _ = enron.generate_prompts(format="prefix-50")
# Replace api_key with your own API key
llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="")
attack = Jailbreak()
results = attack.execute_attack(prompts, llm)
print("results:", results)

