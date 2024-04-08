import os
import pandas as pd

class LegalDocuments:
    
    def __init__(self):
        self.data = []
        #self.path = "./data/dataset/en/"
        #self.path = "./data/dataset/OPP-115/sanitized_policies/"
        #self.path = "./data/dataset/sample/2016/"

        self.path = "./data/dataset/finer139/"
        self.files = [file for file in os.listdir(self.path) if os.path.isfile(self.path + file)]

    def load_train_data(self):
        """for file in self.files:
            with open (self.path + file, "r", encoding="utf-8") as myfile:
                self.data.append(myfile.read())"""

        #df = pd.DataFrame(self.data)
        df = pd.read_json(self.path + "train.jsonl", lines=True)

        return (df.tokens, df.ner_tags)

    def load_test_data(self):

        df = pd.read_json(self.path + "test.jsonl", lines=True)

        return (df.tokens, df.ner_tags)
    
    def load_valid_data(self):

        df = pd.read_json(self.path + "validation.jsonl", lines=True)

        return (df.tokens, df.ner_tags)