from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ClinicalMosaic():
    def __init__(self):
        model = 'Sifal/ClinicalMosaic'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=3,
            torch_dtype='auto',
            trust_remote_code=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        print(self.model.device)
        self.id2label = {0: "entailment", 1: "contradiction", 2: "neutral"}

    def process_data(self, document, sentence):
        inputs = self.tokenizer(document, sentence, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return self.id2label[predicted_class_id]
