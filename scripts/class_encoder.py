import pandas as pd

class ClassEncoder:
    def __init__(self, annot_fp, col, offset=1):
        self.annot_df = pd.read_csv(annot_fp)
        uniques = set(self.annot_df[col])

        self.holder = {}
        self.holder_inverse = {}
        for i, label in enumerate(sorted(uniques)):
            self.holder[label] = i + offset
            self.holder_inverse[i+offset] = label
            
    def transform(self, labels):
        return [self.holder[label] for label in labels]
        
    def inverse_transform(self, labels):
        return [self.holder_inverse[label] for label in labels]