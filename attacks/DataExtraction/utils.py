import pickle
import jsonlines

def load_pickle(filename):
    with open(filename, "rb") as input:
        results = pickle.load(input)
    return results

def load_jsonl(filename):
    results = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            results.append(obj)
    return results