import torch
import numpy as np
import pandas as pd
from dataset import CustomDataset, NERDocuments
from dataset_simpler import CustomSimplerDataset, NERSimplerDocuments
import baseline_lstm_model
import lstm_mha_attn_model
import bilstm_model
import bilstmcrf_model
import matplotlib.pyplot as plt
from logger import Logger
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as c_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score as f1
import copy
import wandb
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import itertools
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

# os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128' # Enable to log runs on Hyperion

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

def save_model(model, epoch):
    print('saving model')
    with open(f'./checkpoints/saved_model_{model.name}_ep_{epoch}.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model(dir):
    print(f'loading model')
    with open(dir, 'rb') as file:
        model = pickle.load(file)

    return model

def load_data(dataset):
    
    return (dataset.load_train_data(), dataset.load_test_data(), dataset.load_valid_data())

def train(model, training_loader, validation_loader, loss_function, optimiser, vocab, indexed_labels, logger, epochs = 5, use_attn = False):
    
    model.to(device)
    loss_function.to(device)

    average_losses = []
    average_valid_f1_scores = []
    average_valid_accuracies = []
    average_valid_losses = []

    model.train()

    for epoch in range(epochs):
        
        print(f'Epoch: {epoch}')

        epoch_losses = []
        valid_f1_scores = []
        valid_accuracies = []
        valid_losses = []
        
        for idx, batch in enumerate(training_loader):
            
            #print(len(batch["tokens"]))

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(training_loader.dataset)/TRAIN_BATCH_SIZE)}')

            tokens = batch["tokens"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"].to(torch.bool)

            model.zero_grad()

            #inputs = convert_to_tensor(tokens[idx], vocab)
            #targets = convert_to_tensor(labels[idx], indexed_labels)

            tokens = tokens.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # predictions, logits = model(tokens, attention_mask, labels)
            # loss = model.calculate_loss(tokens, labels)
            logits = model(tokens, attention_mask, labels)

            # predictions = predictions.to(device)
            # logits = logits.to(device)

            #loss = loss_function(predictions, targets)
            #loss = torch.nn.functional.cross_entropy(predictions, labels)
            loss = loss_function(logits, labels)
            loss.backward()

            optimiser.step()
            epoch_losses.append(np.float64(loss.cpu().detach().numpy()))
            
########################################### VALIDATION ##############################################################

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
                    valid_attention_mask = batch["attention_mask"].to(torch.bool)

                    #print(f'Valid Targets: {valid_targets.tolist()}')

                    valid_inputs = valid_inputs.to(device)
                    valid_targets = valid_targets.to(device)
            
                    # valid_predictions, valid_logits = model(valid_inputs, valid_attention_mask, valid_targets)
                    # valid_loss = model.calculate_loss(valid_inputs, valid_targets)
                    valid_logits = model(valid_inputs, valid_attention_mask, valid_targets)

                    # valid_predictions = valid_predictions.to(device)
                    # valid_logits = valid_logits.to(device)
                    # valid_attention_mask = valid_attention_mask.to(device)

                    valid_loss = loss_function(valid_logits, labels)

                    # valid_predictions = torch.nn.functional.softmax(valid_loss).to(device)

                    softmax = torch.nn.Softmax(dim=0)
                    valid_predictions = softmax(valid_logits)

                    test_pred = torch.argmax(valid_predictions, dim = 1)

                    flattened_targets = valid_targets.view(-1)
                    active_accuracy = valid_attention_mask.view(-1)

                    # print(valid_targets.shape)
                    # print(valid_attention_mask.shape)

                    # active_logits = valid_logits.view(-1, 40)
                    # flattened_predictions = torch.argmax(active_logits, axis=1)

                    # valid_targets = torch.masked_select(valid_targets.view(-1), valid_attention_mask.view(-1))
                    # test_pred = torch.masked_select(torch.argmax((valid_logits.view(-1, 170)), axis = 1), valid_attention_mask.view(-1))

                    # test_targets = flattened_targets
                    # test_pred = flattened_predictions

                    # test_targets = torch.masked_select(flattened_targets, active_accuracy)
                    # test_pred = torch.masked_select(flattened_predictions, active_accuracy)
                    
                    # print(test)
                    # print(valid_targets)

                    # print(len(valid_predictions[0]))
                    # print(len(valid_targets[0]))

                    # acc = torch.eq(valid_set, valid_targets.tolist())
                    # valid_accuracry = (len(set(valid_set).intersection(valid_targets.tolist()))) / len(valid_set)

                    # valid_accuracy = torch.sum(torch.eq(test_pred, valid_targets)).item()/test_pred.nelement()
                    # # print(f'Acc: {valid_accuracry}')
                    # valid_accuracies.append(valid_accuracy)


                    f1_score = f1(valid_targets.flatten().cpu().detach().numpy(), test_pred.flatten().cpu().detach().numpy(), average='weighted')
                    valid_accuracy = torch.sum(torch.eq(test_pred, valid_targets)).item()/test_pred.nelement()

                    valid_f1_scores.append(f1_score)
                    valid_accuracies.append(valid_accuracy)
                    valid_losses.append(np.float64(valid_loss.cpu().detach().numpy()))

                    # if logger != '':
                    #     logger.log({'batch_accuracy': valid_accuracy})
            
            model.train()


        average_losses.append(np.average(epoch_losses))
        average_valid_f1_scores.append(np.average(valid_f1_scores))
        average_valid_accuracies.append(np.average(valid_accuracies))
        average_valid_losses.append(np.average(valid_losses))

        #logger.log({'average_losses': average_losses})
        #logger.log({'average_valid_accuracy': average_valid_accuracy})

        if logger != '':
            logger.log({'train_loss': np.sum(epoch_losses) / len(epoch_losses),
                        'validation_f1_score': np.sum(valid_f1_scores) / len(valid_f1_scores),
                        'validation_accuracy': np.sum(valid_accuracies) / len(valid_accuracies),
                        'validation_loss': np.sum(valid_losses) / len(valid_losses)})

        if epoch % 4 == 0:
            save_model(model, epoch)

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

    save_model(model, epochs)

    print(f'Validation F1 Scores: {average_valid_f1_scores}')
    return average_losses, average_valid_f1_scores

def test(model, testing_loader, ids_to_labels, logger):
    with torch.no_grad():
        max_idx = []
        accuracies = []

        y_true = []
        y_pred = []
        f1_scores = []

        model.eval()

        valid_set = []

        print(f'Testing {int(len(testing_loader.dataset)/VALID_BATCH_SIZE)} batches')

        for idx, batch in enumerate(testing_loader):
            valid_set[:] = []
            
            #inputs = convert_to_tensor(batch[0][idx], indexed_tokens)
            #targets = convert_to_tensor(batch[1][idx], labels_to_id)

            inputs = batch["tokens"]
            targets = batch["labels"]
            attention_mask = batch["attention_mask"]

            #print(valid_targets)

            inputs = inputs.to(device)
            targets = targets.to(device)
    
            logits = model(inputs, attention_mask, targets)

            # predictions = predictions.to(device)

            # predictions = torch.nn.functional.softmax(logits, dim=1).to(device)

            """for prediction in predictions:
                valid_set.append(np.int64(torch.argmax(prediction).cpu()))"""

            softmax = torch.nn.Softmax(dim=0)
            predictions = softmax(logits)

            test_predictions = torch.argmax(predictions, dim = 1)

            f1_score = f1(targets.flatten().cpu().detach().numpy(), test_predictions.flatten().cpu().detach().numpy(), average='weighted')

            #print(len(valid_set))
            y_true.append(targets)
            y_pred.append(test_predictions)
            f1_scores.append(f1_score)
            
            #accuracy = (len(set(valid_set).intersection(targets.tolist()))) / len(valid_set)

            accuracy = torch.sum(torch.eq(test_predictions, targets)).item()/test_predictions.nelement()

            #print(f'Acc: {valid_accuracry}')
            accuracies.append(accuracy)
    
    if logger != '':
        logger.log({'test_accuracy': np.sum(accuracies) / len(accuracies)})

    print(np.sum(f1_scores) / len(f1_scores))

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

    print()
    matrix = confusion_matrix(list(itertools.chain.from_iterable(y_true_labels[0])), list(itertools.chain.from_iterable(y_pred_labels[0])), labels=list(ids_to_labels.values()))

    cm = ConfusionMatrixDisplay(matrix/np.sum(matrix), display_labels=list(ids_to_labels.values()))

    fig, ax = plt.subplots(figsize=(10, 10))
    cm.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()

    report = classification_report(y_true_labels[0], y_pred_labels[0], output_dict = True, zero_division = 0)
    print(f'classification_report:\n{classification_report(y_true_labels[0], y_pred_labels[0], output_dict = False, zero_division = 0)}')

    plt.figure(figsize = (30, 15))
    ax = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, cmap = 'coolwarm', annot=True)
    plt.tight_layout()
    plt.show()
    plt.close()

    if logger != '':
        logger.log({'classification_report': wandb.Image(ax.figure)})
    
    return accuracies

losses = []
valid_f1_scores = []

embedding_dim = 100
hidden_dim = 100

train_sample_frac = 0.25
valid_sample_frac = 0.25
test_sample_frac = 1
epochs = 50
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

logger = ''
wandb_logger = Logger(f"inm706_cw_baseline_lstm_sgd_test", project='inm706_cw_simpler_dataset')
logger = wandb_logger.get_logger()

# dataset = NERDocuments()
dataset = NERSimplerDocuments()
vocab = dataset.get_vocab()
labels_to_id = dataset.get_labels_to_id()
ids_to_labels = dict(map(reversed, labels_to_id.items()))

train_data, test_data, valid_data = load_data(dataset)

# training_set = CustomDataset(train_data, labels_to_id, vocab, train_sample_frac)
# testing_set = CustomDataset(test_data, labels_to_id, vocab, test_sample_frac)
# validation_set = CustomDataset(valid_data, labels_to_id, vocab, valid_sample_frac)

training_set = CustomSimplerDataset(train_data, labels_to_id, vocab, train_sample_frac)
testing_set = CustomSimplerDataset(test_data, labels_to_id, vocab, test_sample_frac)
validation_set = CustomSimplerDataset(valid_data, labels_to_id, vocab, valid_sample_frac)

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

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)

print(f'Dataset len: {len(training_loader.dataset)}')

tokens_to_id = {ch:i for i,ch in enumerate(vocab)}
print(f'Unique Tokens:{len(tokens_to_id)}')

model = baseline_lstm_model.LSTMModel('LSTM_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)
# model = lstm_mha_attn_model.LSTMAttnModel('LSTM_attn_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), tokens_to_id, device)
# model = bilstmcrf_model.BiLSTMCRFModel('BILSTMCRF_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)
# model = bilstm_model.BiLSTMModel('BILSTM_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=labels_to_id['[PAD]'])
optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.009)
# optimiser = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay = 0.0)

# checkpoint_dir = './checkpoints/simpler_dataset/baseline_lstm_1/saved_model_ep_10.pkl'
# model = load_model(checkpoint_dir)

losses, valid_f1_scores = train(model, training_loader, validation_loader, loss_function, optimiser, vocab, labels_to_id, logger, epochs)

# checkpoint_dir = './checkpoints/simpler_dataset/bilstm_test_1/saved_model_BILSTM_test_ep_50.pkl'
# model = load_model(checkpoint_dir)

test_accuracies = test(model, testing_loader, ids_to_labels, logger)
print(f'Test Accuracies: {test_accuracies}')

#plot_loss(1000, test_accuracies, 'test_accuracy')