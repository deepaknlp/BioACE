import transformers
import torch

class Llama_3():
    def __init__(self, prompt=""):
        model_id = "meta-llama/Llama-3.3-70B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )


    def process_data(self, document, sentence):
        if type(document) is list:
            messages = []
            
            for s in sentence:
                for d in document:
                    prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'attributable' or 'not attributable'.\n###Input:\nClaim: " + s + "\n\nReference: " + d + "\n\n### Output:"
                    messages.append([
                        {"role": "user", "content": prompt},
                    ])

            self.pipeline.tokenizer.padding = True
            self.pipeline.tokenizer.padding_side = 'left'

            outputs = self.pipeline(
                messages,
                max_new_tokens=16,
                do_sample=False,
                batch_size=8,
            )

            return [output[0]["generated_text"][-1]["content"] for output in outputs]

        else:
            # prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'attributable' or 'not attributable'.\n###Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Output:"
            # prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'support', 'contradict', or 'neutral'.\n###Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Output:"
            prompt = "For the following lists of answer and document nuggets, select one of the following labels:\n\nSupports: There is at least one document nugget that supports/agrees with at least answer nugget and there are no document nuggets that contradict any answer nuggets.\nContradicts: There is at least one document nugget that disagrees with an answer nugget or states its opposite.\nNeutral: The document nuggets are topically relevant, but lack any information to validate or invalidate any of the answer nuggets.\nNot relevant: The document nuggets are not relevant to the answer nuggets.\nThe response should only include the label.\n\nAnswer Nuggets:\n[sentence]\n\nDocument Nuggets:\n[document]"
            prompt = prompt.replace("[document]", document).replace("[sentence]", sentence)
            messages = [
                {"role": "user", "content": prompt},
            ]

            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
            )
            
            return outputs[0]["generated_text"][-1]["content"]
