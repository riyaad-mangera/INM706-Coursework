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

    def __init__(self, embedding_dim, hidden_dim, vocab_size, labels_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        #self.linear = nn.Linear(hidden_dim, labels_size)

        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim), 
            nn.LSTM(embedding_dim, hidden_dim), 
            nn.Linear(hidden_dim, labels_size)
            )

    def forward(self, input):

        embeddings = self.model[0](input)
        #out_lstm, _ = self.model[1](embeddings.view(len(input), 1, -1))
        out_lstm, _ = self.model[1](embeddings)
        #out_linear = self.model[2](out_lstm.view(len(input), -1))
        out_linear = self.model[2](out_lstm)
        predictions = nn.functional.log_softmax(out_linear, dim=1)

        return predictions