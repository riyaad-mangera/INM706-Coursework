#import nltk
import torch
import torch.nn as nn

class InitialModel:
    
    def __init__(self):
        
        model = nn.Sequential()

        model.add(nn.Embedding())
        model.add(nn.LSTM())
        
        pass

    def word_tokeniser(self, sentence, labels, tokensier):
        pass

class LSTMModelTest():

    def __init__(self, hidden_dim = 0, embedding_dim = 0, tagset_size = 0, embeddings_lambda = 0):
        
        self.hidden_dim = hidden_dim
        self.word_embeddings = embeddings_lambda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
    
class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, labels_size, token_to_id, device):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        #self.linear = nn.Linear(hidden_dim, labels_size)

        """self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, padding_idx = token_to_id["[PAD]"], device = device),
            nn.MultiheadAttention(embedding_dim, num_heads = 2, device = device), 
            nn.LSTM(embedding_dim, hidden_dim, device = device), 
            nn.Linear(hidden_dim, labels_size, device = device)
            )"""
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = token_to_id["[PAD]"], device = device)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads = 2, device = device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, device = device)
        self.linear = nn.Linear(hidden_dim, labels_size, device = device)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, attention_mask, target):

        #print(input.shape)

        #embeddings = self.model[0](input)
        embeddings = self.embedding(input)

        #print(embeddings.shape)

        #out_attention, attn_weights = self.model[1](embeddings.view(len(input), 170, -1), embeddings.view(len(input), 170, -1), embeddings.view(len(input), 170, -1))
        out_attention, attn_weights = self.attention(embeddings.view(len(input), 170, -1), embeddings.view(len(input), 170, -1), embeddings.view(len(input), 170, -1))

        #print(out_attention.shape)

        #out_lstm, _ = self.model[1](embeddings.view(len(input), 1, -1))
        #out_lstm, _ = self.model[2](out_attention.view(len(input), 170, -1))
        out_lstm, _ = self.lstm(out_attention.view(len(input), 170, -1))

        #print(out_lstm.shape)

        #out_linear = self.model[2](out_lstm.view(len(input), -1))
        #out_linear = self.model[3](out_lstm.view(len(input), 170, -1))
        out_linear = self.linear(out_lstm.view(len(input), 170, -1))

        #predictions = nn.functional.log_softmax(out_linear, dim=1)
        predictions = self.softmax(out_linear)

        return predictions, out_linear