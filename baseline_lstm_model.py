import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, name, embedding_dim, hidden_dim, vocab_size, labels_size, label_to_id, device):
        super(LSTMModel, self).__init__()
        
        self.name = name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.label_to_id = label_to_id
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx = self.label_to_id["[PAD]"], device = device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, device = device)
        self.linear = nn.Linear(self.hidden_dim, self.labels_size, device = device)

    def forward(self, input, attention_mask, target):

        # print(f'Input: {input.shape}')

        embeddings = self.embedding(input)

        # print(f'Embed: {embeddings.shape}')

        out_lstm, _ = self.lstm(embeddings)

        # print(f'LSTM: {out_lstm.shape}')

        logits = self.linear(out_lstm)

        # print(f'Linear: {logits.shape}')

        return logits.view(len(input), self.labels_size, -1)