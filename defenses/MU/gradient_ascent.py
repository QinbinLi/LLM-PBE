import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

class GradientAscent:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def inverted_loss(self, logits, labels):
        """
        Define a loss function that rewards incorrect predictions.
        This is the negative of the typical cross-entropy loss for language models.
        """
        loss_fct = torch.nn.CrossEntropyLoss()
        return -loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def tune(self, model, data, learning_rate=1e-4, steps=100):
        """
        Fine-tune the model using gradient ascent on the given data.
        """
        model.train()
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        inputs = self.tokenizer(data['text'], return_tensors="pt").to(self.device)
        labels = torch.tensor([self.tokenizer.encode(data['label'])]).to(self.device)

        for step in range(steps):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = self.inverted_loss(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            print(f"Step {step + 1}: Loss = {loss.item()}")

        return model
