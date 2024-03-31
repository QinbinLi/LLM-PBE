class PromptExtraction:
    def __init__(self):
        pass
                    
    def execute_attack(self, data, model):
        results = []
        for prompt in data:
            results.append(model.query(prompt))
        return results