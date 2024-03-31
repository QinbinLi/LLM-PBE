from data import JailbreakQueries
from models import ChatGPT
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries(people=["Joe Biden"])
llm = ChatGPT(model="gpt-4-0125-preview", api_key="sk-xxx")
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results)
