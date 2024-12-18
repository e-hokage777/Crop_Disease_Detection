import pandas as pd

class ClassEncoder:
    def __init__(self, annot_fp, col, offset=1):
        self.annot_df = pd.read_csv(annot_fp)
        uniques = set(self.annot_df[col])

        self.class_to_idx = {}
        self.idx_to_class = {}
        for i, label in enumerate(sorted(uniques)):
            self.class_to_idx[label] = i + offset
            self.idx_to_class[i+offset] = label
            
    def transform(self, labels):
        return [self.class_to_idx[label] for label in labels]
        
    def inverse_transform(self, labels):
        return [self.idx_to_class[label] for label in labels]