import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):

    def __init__(self, name, embedding_dim, hidden_dim, vocab_size, labels_size, label_to_id, device):
        super(BiLSTMModel, self).__init__()
        
        self.name = name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.label_to_id = label_to_id
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, device=device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, device=device)
        self.linear = nn.Linear(self.hidden_dim * 2, self.labels_size, device=device)
        
    def forward(self, input, attention_mask, target):

        embeddings = self.embedding(input)
        out_lstm, _ = self.lstm(embeddings)
        logits = self.linear(out_lstm)

        return logits.view(len(input), self.labels_size, -1)