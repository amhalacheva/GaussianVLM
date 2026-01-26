import evaluate


class Meteor:

    def __init__(self):
        self.doc = "Meteor with HF implementation"
        self.meteor_hf = evaluate.load('meteor')
        

    def compute_score(self, gts, res):
        res = [val[0] for val in res.values()]
        return [self.meteor_hf.compute(predictions=res, references=list(gts.values()))["meteor"]]