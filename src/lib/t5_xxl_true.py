from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class T5_XXL_TRUE():
    def __init__(self):
        model = "google/t5_xxl_true_nli_mixture"
        self.prompt = "premise: PREMISE_TEXT hypothesis: HYPOTHESIS_TEXT"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def process_data(self, document, sentence):
        prompt = self.prompt.replace("PREMISE_TEXT", document).replace("HYPOTHESIS_TEXT", sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False).to("cuda")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]