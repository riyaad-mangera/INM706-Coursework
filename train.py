import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from dataset import CustomDataset, NERDocuments
import models
import matplotlib.pyplot as plt
from logger import Logger
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as c_report
import copy
import wandb
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

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

def train(model, training_loader, validation_loader, loss_function, optimiser, vocab, indexed_labels, logger, steps, epochs = 5):

    #(tokens, labels) = training_loader[0]

    #print(tokens)
    #print(labels)

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

    #print(indexed_labels)
    #print(len(tokens), len(labels), len(vocab))

    """model = models.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(token_to_ix), len(label_to_ix))
    loss_function = torch.nn.NLLLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)"""
    
    model.to(device)
    loss_function.to(device)

    average_losses = []
    average_valid_accuracy = []
    average_valid_losses = []

    model.train()

    for epoch in range(epochs):
        
        print(f'Epoch: {epoch}')

        epoch_losses = []
        valid_accuracies = []
        valid_losses = []
        
        for idx, batch in enumerate(training_loader):
            
            #print(len(batch["tokens"]))

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(training_loader.dataset)/TRAIN_BATCH_SIZE)}')

            tokens = batch["tokens"]
            labels = batch["labels"]

            model.zero_grad()

            #inputs = convert_to_tensor(tokens[idx], vocab)
            #targets = convert_to_tensor(labels[idx], indexed_labels)

            tokens = tokens.to(device)
            labels = labels.to(device)
            
            predictions = model(tokens)

            predictions.to(device)

            #loss = loss_function(predictions, targets)
            loss = torch.nn.functional.cross_entropy(predictions, labels)
            loss.backward()

            optimiser.step()
            epoch_losses.append(np.float64(loss.cpu().detach().numpy()))
            
            with torch.no_grad():
                model.eval()

                max_idx = []

                #(valid_tokens, valid_labels) = batch[1]

                print(f'\tValidating {int(len(validation_loader.dataset)/VALID_BATCH_SIZE)} Batches')
                
                for index, batch in enumerate(validation_loader):
                    
                    #valid_inputs = convert_to_tensor(batch[0][idx], vocab)
                    #valid_targets = convert_to_tensor(batch[1][idx], indexed_labels)

                    valid_inputs = batch["tokens"]
                    valid_targets = batch["labels"]

                    #print(f'Valid Targets: {valid_targets.tolist()}')

                    valid_inputs = valid_inputs.to(device)
                    valid_targets = valid_targets.to(device)
            
                    valid_predictions = model(valid_inputs)

                    valid_predictions.to(device)

                    valid_loss = torch.nn.functional.cross_entropy(valid_predictions, valid_targets)

                    #print(f'Valid Set: {valid_set}')

                    test_pred = torch.argmax(valid_predictions, dim = 1)

                    #print(test)
                    #print(valid_targets)

                    #print(len(valid_predictions[0]))
                    #print(len(valid_targets[0]))

                    #acc = torch.eq(valid_set, valid_targets.tolist())
                    #valid_accuracry = (len(set(valid_set).intersection(valid_targets.tolist()))) / len(valid_set)

                    valid_accuracy = torch.sum(torch.eq(test_pred, valid_targets)).item()/test_pred.nelement()
                    #print(f'Acc: {valid_accuracry}')
                    valid_accuracies.append(valid_accuracy)
                    valid_losses.append(np.float64(valid_loss.cpu().detach().numpy()))

                    logger.log({'batch_accuracy': valid_accuracy})
            
            model.train()


        average_losses.append(np.average(epoch_losses))
        average_valid_accuracy.append(np.average(valid_accuracies))
        average_valid_losses.append(np.average(valid_losses))

        #logger.log({'average_losses': average_losses})
        #logger.log({'average_valid_accuracy': average_valid_accuracy})

        logger.log({'train_loss': np.sum(epoch_losses) / len(epoch_losses)})
        logger.log({'validation_accuracy': np.sum(valid_accuracies) / len(valid_accuracies)})
        logger.log({'validation_loss': np.sum(valid_losses) / len(valid_losses)})

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

    print(f'Validation Accuracies: {average_valid_accuracy}')
    return average_losses, average_valid_accuracy

def test(model, testing_loader, ids_to_labels, logger):
    with torch.no_grad():
        max_idx = []
        accuracies = []

        y_true = []
        y_pred = []

        model.eval()

        valid_set = []

        for idx, batch in enumerate(testing_loader):
            valid_set[:] = []
            
            #inputs = convert_to_tensor(batch[0][idx], indexed_tokens)
            #targets = convert_to_tensor(batch[1][idx], labels_to_id)

            inputs = batch["tokens"]
            targets = batch["labels"]

            #print(valid_targets)

            inputs = inputs.to(device)
            targets = targets.to(device)
    
            predictions = model(inputs)

            predictions.to(device)

            """for prediction in predictions:
                valid_set.append(np.int64(torch.argmax(prediction).cpu()))"""

            test_predictions = torch.argmax(predictions, dim = 1)

            #print(len(valid_set))
            y_true.append(targets)
            y_pred.append(test_predictions)
            
            #accuracy = (len(set(valid_set).intersection(targets.tolist()))) / len(valid_set)

            accuracy = torch.sum(torch.eq(test_predictions, targets)).item()/test_predictions.nelement()

            #print(f'Acc: {valid_accuracry}')
            accuracies.append(accuracy)
    
    logger.log({'test_accuracy': np.sum(accuracies) / len(accuracies)})

    #print(f'y_true: {y_true}')
    #print(f'y_pred: {y_pred}')

    y_true_labels = []
    y_pred_labels = []

    #print(len(y_true[0]), len(y_pred[0]))

    for labels in y_true:
        #print(labels[0][0])
        y_true_labels.append([[ids_to_labels.get(np.int64(label.cpu().item())) for label in tens_labels] for tens_labels in labels])

    #print(y_true_labels)

    for labels in y_pred:
        #print(labels[0][0])
        y_pred_labels.append([[ids_to_labels.get(np.int64(label.cpu().item())) for label in tens_labels] for tens_labels in labels])

    #print(y_pred_labels)

    """for labels in y_true:
        
        y_labels.append([list(labels_to_id.keys())[list(labels_to_id.values()).index(idx)] for idx in labels])

    for labels in y_pred:

        y_predictions.append([list(labels_to_id.keys())[list(labels_to_id.values()).index(idx)] for idx in labels])

    print(len(y_labels), len(y_predictions))"""

    #y_labels = [list(labels_to_id.keys())[list(labels_to_id.values()).index(idx)] for idx in y_true]
    #y_predictions = [list(labels_to_id.keys())[list(labels_to_id.values()).index(idx)] for idx in y_pred]

    #y_labels = [labels_to_id[id] for id in y_true]
    #y_predictions = [labels_to_id[id] for id in y_pred]

    #test_report = c_report(y_true, y_pred)

    #print(test_report)

    report = classification_report(y_true_labels[0], y_pred_labels[0], output_dict = True)
    print(f'classification_report:\n{classification_report(y_true_labels[0], y_pred_labels[0], output_dict = False)}')

    plt.figure(figsize = (15, 30))
    ax = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, cmap = 'coolwarm', annot=True)
    plt.tight_layout()

    logger.log({'classification_report': wandb.Image(ax.figure)})
    

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

embedding_dim = 100
hidden_dim = 100

steps = 10000
epochs = 20
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

wandb_logger = Logger(f"inm705_cw_test_lstm", project='inm705_cw')
logger = wandb_logger.get_logger()
#logger = []

dataset = NERDocuments()
vocab = dataset.get_vocab()
labels_to_id = dataset.get_labels_to_id()
ids_to_labels = dict(map(reversed, labels_to_id.items()))

train_data, test_data, valid_data = load_data(dataset)

training_set = CustomDataset(train_data, labels_to_id, vocab, 16384)
testing_set = CustomDataset(test_data, labels_to_id, vocab, 16384)
validation_set = CustomDataset(valid_data, labels_to_id, vocab, 512)

"""flat_labels = [label for labels in train_data[1] for label in labels]
unique_labels = list(set(flat_labels))

labels_to_id = dict(zip(unique_labels, [idx for idx in range(len(unique_labels))]))"""

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

"""print(len(train_data[2]), len(train_data[3]))
print(torch.stack(train_data[2]).size())

#training_set = {"tokens": train_data[2], "labels": train_data[3]}
training_set = TensorDataset(torch.stack(train_data[2]), torch.stack(train_data[3]))
testing_set = TensorDataset(torch.stack(test_data[2]), torch.stack(test_data[3]))
validation_set = TensorDataset(torch.stack(valid_data[2]), torch.stack(valid_data[3]))"""

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)

print(f'Dataset len: {len(training_loader.dataset)}')

"""training_set = {"targets": train_data[0], "labels": train_data[1]}
testing_set = {"targets": test_data[0], "labels": test_data[1]}
validation_set = {"targets": valid_data[0], "labels": valid_data[1]}"""

model = models.LSTMModel(embedding_dim, hidden_dim, len(vocab), len(labels_to_id))
loss_function = torch.nn.CrossEntropyLoss()
#optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
optimiser = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay = 0.0)

losses, valid_accuracies = train(model, training_loader, validation_loader, loss_function, optimiser, vocab, labels_to_id, logger, steps, epochs)

#print(losses)

#plot_loss(epochs, losses, 'losses')
#plot_loss(epochs, valid_accuracies, 'accuracy')

test_accuracies = test(model, testing_loader, ids_to_labels, logger)
print(f'Test Accuracies: {test_accuracies}')

#plot_loss(1000, test_accuracies, 'test_accuracy')