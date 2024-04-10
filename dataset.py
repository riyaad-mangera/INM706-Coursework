import os
import pandas as pd

class NERDocuments:
    
    def __init__(self):
        self.data = []
        #self.path = "./data/dataset/en/"
        #self.path = "./data/dataset/OPP-115/sanitized_policies/"
        #self.path = "./data/dataset/sample/2016/"

        self.path = "./data/dataset/finer139/"
        self.files = [file for file in os.listdir(self.path) if os.path.isfile(self.path + file)]

        self.vocab = {}

    def preprocess_data(self, tokens, labels):

        tokens = tokens.tolist()
        labels = labels.tolist()

        #tokens = ["[CLS]"] + tokens + ["[SEP]"]

        """tokens.insert(0, "[CLS]")
        tokens.insert(-1, "[SEP]")

        labels.insert(0, "O")
        labels.insert(-1, "O")"""

        """print(pd.concat([pd.Series("[CLS]"), tokens, pd.Series("[SEP]")], axis = 0, ignore_index = False))
        pd.concat([pd.Series("0"), labels, pd.Series("0")], axis = 0)"""

        """tokens[0] = "[CLS]"
        tokens[-1] = "[SEP]"

        labels[0] = "O"
        labels[-1] = "O"""

        return (tokens, labels)

    def add_to_vocab(self, data_tokens):
        flat_tokens = [token for tokens in data_tokens for token in tokens]
        
        for token in flat_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def load_train_data(self):
        """for file in self.files:
            with open (self.path + file, "r", encoding="utf-8") as myfile:
                self.data.append(myfile.read())"""

        #df = pd.DataFrame(self.data)
        df = pd.read_json(self.path + "train.jsonl", lines=True)

        #return (df.tokens, df.ner_tags)

        self.add_to_vocab(df.tokens)

        return self.preprocess_data(df.tokens, df.ner_tags)

    def load_test_data(self):

        df = pd.read_json(self.path + "test.jsonl", lines=True)

        self.add_to_vocab(df.tokens)

        return self.preprocess_data(df.tokens, df.ner_tags)
    
    def load_valid_data(self):

        df = pd.read_json(self.path + "validation.jsonl", lines=True)

        self.add_to_vocab(df.tokens)

        return self.preprocess_data(df.tokens, df.ner_tags)
    
    def get_vocab(self):
        return self.vocab
    
