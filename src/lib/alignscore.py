from alignscore import AlignScore

class AlignScorer():
    def __init__(self):
        self.scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='data/AlignScore-large.ckpt', evaluation_mode='nli_sp')

    def process_data(self, document, sentence):
        return self.scorer.score(contexts=[document], claims=[sentence])[0]