from data.jailbreakqueries import JailbreakQueries
# from models.togetherai import TogetherAIModels
from attacks.Jailbreak.jailbreak import Jailbreak
from models.hf_models import HFModels
import json
models = [
    # "EleutherAI/pythia-14m",
    # "EleutherAI/pythia-31m",
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    ]
people = ["Mark Zuckerberg", "Tim Cook", "Elon Musk", "Joe Biden", "Donald Trump", "Barack Obama", "Dawn Song", "Yann LeCun", "Los Angeles"]
data = JailbreakQueries(people=people)
queries = data.generate_queries()
for model_name in models:
    llm = HFModels(model_name=model_name, max_length=500)
    attack = Jailbreak()
    try:
        results = attack.execute_attack(queries, llm)
    except:
        continue
    with open('results_'+model_name.split("/")[-1]+'.json', 'w') as f:
        json.dump(results, f)
