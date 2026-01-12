from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FLAN_T5():
    def __init__(self, prompt=""):
        # model = "google/flan-t5-xxl"
        model = "./data/finetuned_models/flan-t5-xxl/checkpoint-200"
        self.prompt = prompt
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def process_data(self, document, sentence):
        # prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'attributable' or 'not attributable'.\n###Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Output:"
        prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'support', 'contradict', or 'neutral'.\n###Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Output:"
        prompt = prompt.replace("[document]", document).replace("[sentence]", sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]