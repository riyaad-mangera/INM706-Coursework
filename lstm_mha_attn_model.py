import torch
import torch.nn as nn

class LSTMAttnModel(nn.Module):

    def __init__(self, name, embedding_dim, hidden_dim, vocab_size, labels_size, label_to_id, device):
        super(LSTMAttnModel, self).__init__()
        
        self.name = name
        self.hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.label_to_id = label_to_id
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = label_to_id["[PAD]"], device = device)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads = 4, device = device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, device = device)
        self.linear = nn.Linear(hidden_dim, labels_size, device = device)

    def forward(self, input, attention_mask, target):

        embeddings = self.embedding(input)
        out_lstm, _ = self.lstm(embeddings)
        out_attention, attn_weights = self.attention(out_lstm, out_lstm, out_lstm)
        logits = self.linear(out_attention)

        return logits.view(len(input), self.labels_size, -1)