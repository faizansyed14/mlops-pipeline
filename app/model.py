from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        label = int(torch.argmax(logits, dim=1))
        return {"label": label, "probabilities": probs}
