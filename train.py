import torch
import tensorflow as tf
import numpy as np
from dataset import LegalDocuments
import models
import matplotlib.pyplot as plt
from logger import Logger

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

embedding_dim = 100
hidden_dim = 100

def convert_to_tensor(sequence, indexed_var):
    indexes = [indexed_var[item] for item in sequence]
    return torch.tensor(indexes, dtype=torch.long)

def plot_loss(epochs, losses, name = 'fig'):
    plt.plot([x for x in range(epochs)], losses)
    plt.savefig(f'{name}.png')
    plt.clf()
    plt.close()

def load_data(dataset):
    
    return (dataset.load_train_data(), dataset.load_test_data(), dataset.load_valid_data())

def train(model, train_data, valid_data, loss_function, optimiser, indexed_tokens, indexed_labels, logger, steps, epochs = 5):

    (tokens, labels) = train_data

    print(tokens)
    print(labels)

    """flat_tokens = [token for innerList in tokens for token in innerList]
    token_idx = [idx for idx in range(len(flat_tokens))]

    print(len(flat_tokens))
    print(len(token_idx))

    indexed_tokens = {}

    for token in flat_tokens:
        if token not in indexed_tokens:
            indexed_tokens[token] = len(indexed_tokens)"""

    """flat_labels = [label for innerList in labels for label in innerList]
    unique_labels = list(set(flat_labels))

    indexed_labels = dict(zip(unique_labels, [idx for idx in range(len(unique_labels))]))

    flat_valid_tokens = [token for tokens in valid_data[0] for token in tokens]
    valid_token_idx = [idx for idx in range(len(flat_valid_tokens))]

    indexed_valid_tokens = {}

    for token in flat_valid_tokens:
        if token not in indexed_valid_tokens:
            indexed_valid_tokens[token] = len(indexed_valid_tokens)

    flat_valid_labels = [label for innerList in valid_data[1] for label in innerList]
    unique_valid_labels = list(set(flat_valid_labels))

    indexed_valid_labels = dict(zip(unique_valid_labels, [idx for idx in range(len(unique_valid_labels))]))"""

    print(indexed_labels)
    print(len(tokens), len(indexed_tokens))

    """model = models.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(token_to_ix), len(label_to_ix))
    loss_function = torch.nn.NLLLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)"""
    
    model.to(device)
    loss_function.to(device)

    average_losses = []
    average_valid_accuracy = []

    model.train()

    for epoch in range(epochs):
        
        print(f'Epoch: {epoch}')

        epoch_losses = []
        valid_accuracies = []
        
        for idx in range(steps):
            
            print(f'Epoch: {epoch}, Step {idx} of {steps}')

            model.zero_grad()

            inputs = convert_to_tensor(tokens[idx], indexed_tokens)
            targets = convert_to_tensor(labels[idx], indexed_labels)

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs)

            predictions.to(device)

            #loss = loss_function(predictions, targets)
            loss = torch.nn.functional.cross_entropy(predictions, targets)
            loss.backward()

            optimiser.step()
            epoch_losses.append(np.float64(loss.cpu().detach().numpy()))
            
            #with torch.no_grad():
                
            valid_set = []
            max_idx = []

            (valid_tokens, valid_labels) = valid_data
            model.eval()

            for idx in range(10):
                valid_inputs = convert_to_tensor(valid_tokens[idx], indexed_tokens)
                valid_targets = convert_to_tensor(valid_labels[idx], indexed_labels)

                #print(valid_targets)

                valid_inputs = valid_inputs.to(device)
                valid_targets = valid_targets.to(device)
        
                valid_predictions = model(valid_inputs)

                valid_predictions.to(device)

                for prediction in valid_predictions:
                    valid_set.append(torch.argmax(prediction))
                    #print(f'Predictions: {torch.argmax(prediction)}')

                #print(valid_set)

                #acc = torch.eq(valid_set, valid_targets.tolist())
                valid_accuracry = (len(set(valid_set).intersection(valid_targets.tolist()))) / len(valid_set)
                #print(f'Acc: {valid_accuracry}')
                valid_accuracies.append(valid_accuracry)
            
            model.train()


        average_losses.append(np.average(epoch_losses))
        average_valid_accuracy.append(np.average(valid_accuracies))

        #logger.log({'average_losses': average_losses})
        #logger.log({'average_valid_accuracy': average_valid_accuracy})

        logger.log({'train_loss': np.sum(epoch_losses) / len(epoch_losses)})
        logger.log({'validation_accuracy': np.sum(valid_accuracies) / len(valid_accuracies)})

    """with torch.no_grad():
        inputs = convert_to_tensor(tokens[len(tokens) - 2], indexed_tokens)

        inputs = inputs.to(device)

        predictions = model(inputs)

        print(len(predictions))
        print(predictions)

        max_scores = []
        max_idx = []
        tag_scores_list = predictions.tolist()

        for prediction in predictions:
            print(torch.argmax(prediction))

        f = open('predictions.txt', 'w')
        

        for scores in tag_scores_list:
            
            f.write('\n')
            f.write('.'.join(str(score) for score in scores))
        

            print(np.max(scores))
            max_idx.append(scores.index(np.max(scores)))

        print(f'Max Idx:\n{max_idx}')

        f.close()

        test = []

        print(indexed_labels)

        for idx in max_idx:
            
            #print(f'Test:\n{list(indexed_labels.keys())[list(indexed_labels.values()).index(idx)]}')
            test.append(list(indexed_labels.keys())[list(indexed_labels.values()).index(idx)])

        print(tokens[len(tokens) - 2])
        print(test)"""

    print(average_valid_accuracy)
    return average_losses, average_valid_accuracy

def test(model, tokens, labels, indexed_tokens, indexed_labels, logger, steps = 5):
    with torch.no_grad():
        valid_set = []
        max_idx = []
        accuracies = []

        model.eval()

        for idx in range(steps):
            inputs = convert_to_tensor(tokens[idx], indexed_tokens)
            targets = convert_to_tensor(labels[idx], indexed_labels)

            #print(valid_targets)

            inputs = inputs.to(device)
            targets = targets.to(device)
    
            predictions = model(inputs)

            predictions.to(device)

            for prediction in predictions:
                valid_set.append(torch.argmax(prediction))

            accuracry = (len(set(valid_set).intersection(targets.tolist()))) / len(valid_set)
            #print(f'Acc: {valid_accuracry}')
            accuracies.append(accuracry)
    
    logger.log({'test_accuracy': np.sum(accuracies) / len(accuracies)})

    return accuracies

    """with torch.no_grad():
        
        for idx in range(len(tokens)):
        
            inputs = convert_to_tensor(tokens[len(tokens) - 2], token_to_ix)

            inputs = inputs.to(device)

            tag_scores = model(inputs)
            
            print(len(tag_scores))
            print(tag_scores)

            max_scores = []
            max_idx = []
            tag_scores_list = tag_scores.tolist()        

        for scores in tag_scores_list:

            print(np.max(scores))
            max_idx.append(scores.index(np.max(scores)))

        print(max_idx)


        test = []

        for idx in max_idx:
            
            test.append(list(label_to_ix.keys())[list(label_to_ix.values()).index(idx)])

        print(tokens[len(tokens) - 2])
        print(test)"""

losses = []
valid_accuracies = []
steps = 1000
epochs = 50

wandb_logger = Logger(f"inm705_cw_test_lstm", project='inm705_cw')
logger = wandb_logger.get_logger()

dataset = LegalDocuments()

train_data, test_data, valid_data = load_data(LegalDocuments())

flat_tokens = [*[token for tokens in train_data[0] for token in tokens], 
               *[token for tokens in valid_data[0] for token in tokens], 
               *[token for tokens in test_data[0] for token in tokens]]
token_idx = [idx for idx in range(len(flat_tokens))]

indexed_tokens = {}

for token in flat_tokens:
    if token not in indexed_tokens:
        indexed_tokens[token] = len(indexed_tokens)

flat_labels = [label for labels in train_data[1] for label in labels]
unique_labels = list(set(flat_labels))

indexed_labels = dict(zip(unique_labels, [idx for idx in range(len(unique_labels))]))

model = models.LSTMModel(embedding_dim, hidden_dim, len(indexed_tokens), len(indexed_labels))
loss_function = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

losses, valid_accuracies = train(model, train_data, valid_data, loss_function, optimiser, indexed_tokens, indexed_labels, logger, steps, epochs)

#print(losses)

plot_loss(epochs, losses, 'losses')
plot_loss(epochs, valid_accuracies, 'accuracy')

test_accuracies = test(model, test_data[0], test_data[1], indexed_tokens, indexed_labels, logger, steps = 100)

#plot_loss(1000, test_accuracies, 'test_accuracy')