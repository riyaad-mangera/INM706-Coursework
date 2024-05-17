import torch
import torch.nn as nn

class BiLSTMCRFModel(nn.Module):

    def __init__(self, name, embedding_dim, hidden_dim, vocab_size, labels_size, label_to_id, device):
        super(BiLSTMCRFModel, self).__init__()
        
        self.name = name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.label_to_id = label_to_id
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, device=device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, device=device)

        self.linear = nn.Linear(self.hidden_dim, self.labels_size, device=device)

        self.transitions = nn.Parameter(torch.randn(self.labels_size, self.labels_size).to(device)).to(device)

        # print(self.label_to_id)
        self.transitions.data[self.label_to_id["[CLS]"], :] = -10000
        self.transitions.data[:, self.label_to_id["[SEP]"]] = -10000

        self.hidden = (torch.randn(2, 50, self.hidden_dim // 2).to(self.device),
                       torch.randn(2, 50, self.hidden_dim // 2).to(self.device))

    def compute_log_sum(self, score):
        max_score = score[0, torch.argmax(score)]
        max_score_broadcast = max_score.view(1, -1).expand(1, score.size()[1])

        return max_score + torch.log(torch.sum(torch.exp(score - max_score_broadcast)))

    def compute_partition_function(self, logits):
        
        init_alphas = torch.full((1, self.labels_size), -10000.)
        init_alphas[0][self.label_to_id["[CLS]"]] = 0.

        forward_var = init_alphas

        for logit in logits:
            forward_tensors = []
            for next_tag in range(self.labels_size):
                
                emission_score = logit[next_tag].view(1, -1).expand(1, self.labels_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                
                trans_score = trans_score.to(self.device)
                forward_var = forward_var.to(self.device)

                next_tag_var = forward_var + trans_score + emission_score
                
                forward_tensors.append(self.compute_log_sum(next_tag_var).view(1))

            forward_var = torch.cat(forward_tensors).view(1, -1)

        terminal_var = forward_var + self.transitions[self.label_to_id["[SEP]"]]
        forward_score = self.compute_log_sum(terminal_var)

        return forward_score
    
    def compute_sentence_score(self, logits, target):
        
        gold_score = torch.zeros(1).to(self.device)
        target = target.reshape(-1)

        target = torch.cat([torch.tensor([self.label_to_id["[CLS]"]], dtype=torch.long).to(self.device), target])

        for idx, logit in enumerate(logits):
            gold_score = gold_score + self.transitions[target[idx + 1], target[idx]] + logit[target[idx + 1]]

        gold_score = gold_score + self.transitions[self.label_to_id["[SEP]"], target[-1]]
        # print(score)

        return gold_score

    def viterbi_decode(self, features):
        backpointers = []
        print(len(features))

        init_vvars = torch.full((50, self.labels_size), -10000.)
        init_vvars[0][self.label_to_id["[CLS]"]] = 0

        forward_var = init_vvars

        for feature in features:
            timestep_backpointers = []
            viterbivars_t = []

            for next_tag in range(self.labels_size):
                
                forward_var = forward_var.to(self.device)
                next_tag_var = forward_var + self.transitions[next_tag]

                # print(f'Next tag var:\n{next_tag_var}')

                # print(next_tag_var.shape)

                best_tag_id = torch.argmax(next_tag_var)
                timestep_backpointers.append(best_tag_id)

                print(best_tag_id)

                # print(next_tag_var.shape)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # print(f'BEFORE:\n{forward_var.shape}')

            forward_var = (torch.cat(viterbivars_t) + feature).view(50, -1)

            # print(f'AFTER:\n{forward_var.shape}')

            backpointers.append(timestep_backpointers)

        terminal_var = forward_var + self.transitions[self.label_to_id["[SEP]"]]

        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        path_predictions = [best_tag_id]

        for timestep_backpointers in reversed(backpointers):

            best_tag_id = timestep_backpointers[best_tag_id]
            path_predictions.append(best_tag_id)

        start = path_predictions.pop()

        assert start == self.label_to_id["[CLS]"]
        path_predictions.reverse()

        return path_score, path_predictions

    def calculate_loss(self, input, target):

        # print(f'Target: {target.shape}')
        
        self.hidden = (torch.randn(2, 50, self.hidden_dim // 2).to(self.device),
                       torch.randn(2, 50, self.hidden_dim // 2).to(self.device))
        
        # print(f'Hidden: {self.hidden[0].shape}')

        embeddings = self.embeddings(input)
        # print(f'Embeddings: {embeddings.shape}')

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

        logits = self.linear(lstm_out)

        forward_score = self.compute_partition_function(logits)
        gold_score = self.compute_sentence_score(logits, target)

        # print(forward_score - gold_score)

        return torch.max(forward_score - gold_score)

    def forward(self, input, attention_mask, target):

        # print(f'Input: {input.shape}')

        self.hidden = (torch.randn(2, 50, self.hidden_dim // 2).to(self.device),
                       torch.randn(2, 50, self.hidden_dim // 2).to(self.device))

        embeddings = self.embedding(input) #.view(len(input), 1, -1)
        # print(f'Embeddings: {embeddings.shape}')

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # print(f'LSTM: {lstm_out.shape}')

        linear_out = self.linear(lstm_out)

        # print(f'Linear: {linear_out.shape}')

        score, predictions = self.viterbi_decode(linear_out)
        print(len(predictions))

        return score, predictions