import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels_to_id, vocab, slice_ratio):

        self.data = data.head(slice_ratio)

        self.tokens = data.tokens
        self.labels = data.ner_tags

        self.labels_to_id = labels_to_id
        self.ids_to_label = dict(map(reversed, labels_to_id.items()))
        self.vocab = vocab

        #self.data = pd.read_json(self.path + f"{self.file_name}.jsonl", lines=True)

    def convert_to_tensor(self, sequence, indexed_var):
        
        indexes = [indexed_var[item] for item in sequence]

        return torch.tensor(indexes, dtype=torch.long)
    
    def __getitem__(self, index):
        
        tokens_as_lst = self.tokens.tolist()
        labels_as_lst = self.labels.tolist()

        tokens_ids = []
        labels_ids = []

        #print(len(tokens_as_lst), len(labels_as_lst))

        #for token, label in zip(tokens_as_lst, labels_as_lst):

        tokens_as_lst[index] = ["[CLS]"] + tokens_as_lst[index] + ["[SEP]"]
        #labels_as_lst[index].insert(0, "O")
        #labels_as_lst[index].insert(-1, "O")

        labels_as_lst[index] = ["O"] + labels_as_lst[index] + ["O"]

        maxlen = len(self.labels_to_id)
        #maxlen = 128

        if (len(tokens_as_lst[index]) > maxlen):
            # truncate
            tokens_as_lst[index] = tokens_as_lst[index][:maxlen]
            labels_as_lst[index] = labels_as_lst[index][:maxlen]
        else:
            # pad
            tokens_as_lst[index] = tokens_as_lst[index] + ["[PAD]" for _ in range(maxlen - len(tokens_as_lst[index]))]
            labels_as_lst[index] = labels_as_lst[index] + ["O" for _ in range(maxlen - len(labels_as_lst[index]))]

        attention_mask = [1 if token != "[PAD]" else 0 for token in tokens_as_lst]

        token_ids = self.convert_to_tensor(tokens_as_lst[index], self.vocab)
        label_ids = [self.labels_to_id[label] for label in labels_as_lst[index]]

        return {"tokens": torch.tensor(token_ids, dtype = torch.long), 
                "labels": torch.tensor(label_ids, dtype = torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype = torch.long)
                }
    
    def __len__(self):
        return len(self.data)

class NERDocuments:
    
    def __init__(self):
        self.data = []
        #self.path = "./data/dataset/en/"
        #self.path = "./data/dataset/OPP-115/sanitized_policies/"
        #self.path = "./data/dataset/sample/2016/"

        self.path = "./data/dataset/finer139/"
        self.files = [file for file in os.listdir(self.path) if os.path.isfile(self.path + file)]

        self.train_data = pd.read_json(self.path + "train.jsonl", lines=True)
        self.test_data = pd.read_json(self.path + "test.jsonl", lines=True)
        self.valid_data = pd.read_json(self.path + "validation.jsonl", lines=True)

        flat_labels = [*[label for labels in self.train_data.ner_tags for label in labels],
                       *[label for labels in self.test_data.ner_tags for label in labels],
                       *[label for labels in self.valid_data.ner_tags for label in labels]]
        
        #flat_labels = [label for labels in self.train_data.ner_tags for label in labels]

        unique_labels = list(set(flat_labels))
        print(len(unique_labels))

        self.labels_to_id = dict(zip(unique_labels, [idx for idx in range(len(unique_labels))]))

        self.vocab = {}
        self.add_to_vocab(self.train_data.tokens)
        self.add_to_vocab(self.test_data.tokens)
        self.add_to_vocab(self.valid_data.tokens)

        self.vocab['[CLS]'] = len(self.vocab)
        self.vocab['[SEP]'] = len(self.vocab)
        self.vocab['[PAD]'] = len(self.vocab)

        #print(self.labels_to_id)

    def convert_to_tensor(self, sequence, indexed_var):
        indexes = [indexed_var[item] for item in sequence]
        return torch.tensor(indexes, dtype=torch.long)

    def preprocess_data(self, tokens, labels):

        tokens_as_lst = tokens.tolist()
        labels_as_lst = labels.tolist()

        tokens_ids = []
        labels_ids = []

        print(len(tokens_as_lst), len(labels_as_lst))

        for token, label in zip(tokens_as_lst, labels_as_lst):

            token = ["[CLS]"] + token + ["[SEP]"]
            label.insert(0, "O")
            label.insert(-1, "O")

            maxlen = len(self.labels_to_id)

            if (len(token) > maxlen):
                # truncate
                token = token[:maxlen]
                label = label[:maxlen]
            else:
                # pad
                token = token + ["[PAD]" for _ in range(maxlen - len(token))]
                label = label + ["O" for _ in range(maxlen - len(label))]

            token_ids = self.convert_to_tensor(token, self.vocab)
            label_ids = [self.labels_to_id[label] for label in label]

            tokens_ids.append(torch.tensor(token_ids))
            labels_ids.append(torch.tensor(label_ids))

        return (tokens_as_lst, labels_as_lst, tokens_ids, labels_ids)

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
        #df = pd.read_json(self.path + "train.jsonl", lines=True)

        #return (df.tokens, df.ner_tags)

        #self.add_to_vocab(df.tokens)

        #return self.preprocess_data(self.train_data.tokens, self.train_data.ner_tags)
        #return (self.train_data.tokens, self.train_data.ner_tags)
        return self.train_data

    def load_test_data(self):

        #df = pd.read_json(self.path + "test.jsonl", lines=True)

        #self.add_to_vocab(df.tokens)

        #return self.preprocess_data(self.test_data.tokens, self.test_data.ner_tags)
        #return (self.test_data.tokens, self.test_data.ner_tags)
        return self.test_data
    
    def load_valid_data(self):

        #df = pd.read_json(self.path + "validation.jsonl", lines=True)

        #self.add_to_vocab(df.tokens)

        #return self.preprocess_data(self.valid_data.tokens, self.valid_data.ner_tags)
        #return (self.valid_data.tokens, self.valid_data.ner_tags)
        return self.valid_data
    
    def get_vocab(self):
        return self.vocab
    
    def get_labels_to_id(self):
        return self.labels_to_id
    
