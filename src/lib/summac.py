from summac.model_summac import SummaCZS, SummaCConv

class SummaCScore():
    def __init__(self, granularity="sentence", model_name="vitc", device="cuda"):
        self.granularity = granularity
        self.model_name = model_name
        self.device = device
        self.model_zs = SummaCZS(granularity=self.granularity, model_name=self.model_name, device=self.device)
        self.model_conv = SummaCConv(models=[self.model_name], bins='percentile', granularity=self.granularity, nli_labels="e", device=self.device, start_file="default", agg="mean")


    def process_data(self, document, sentence):
        score_zs=self.model_zs.score([document], [sentence])["scores"][0]
        score_conv=self.model_conv.score([document], [sentence])["scores"][0]

        return [score_conv, score_zs]
