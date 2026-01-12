import os
from openai import OpenAI

class T5_XXL_TRUE():
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "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"

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )

    def process_data(self, document, sentence):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-4o",
        )

        return chat_completion.choices[0].message