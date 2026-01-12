from transformers import AutoModelForCausalLM, AutoTokenizer

class AttrScore_Alpaca():
    def __init__(self, prompt=""):
        model="osunlp/attrscore-alpaca-13b"
        self.prompt = prompt
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def process_data(self, document, sentence):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nVerify whether a given reference can support the claim. Options: Attributable, Extrapolatory or Contradictory.\n\n### Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Response:"
        prompt = prompt.replace("[document]", document).replace("[sentence]", sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]