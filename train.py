import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
import pickle
import itertools
import yaml
from torch.utils.data import DataLoader
from logger import Logger

from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score as f1

from warnings import simplefilter

from dataset_simpler import CustomSimplerDataset, NERSimplerDocuments
import baseline_lstm_model
import lstm_mha_attn_model
import bilstm_model
import bilstmcrf_model

simplefilter(action='ignore', category=FutureWarning)

# os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128' # Enable to log runs on Hyperion

#============================ PARAMETER INITIALISATION ===================================================================

with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

config_settings = config['model_settings']

TRAIN_BATCH_SIZE = config_settings['batch_size']
VALID_BATCH_SIZE = config_settings['batch_size']

embedding_dim = config_settings['embedding_dim']
hidden_dim = config_settings['hidden_dim']

train_sample_frac = config_settings['train_sample_frac']
valid_sample_frac = config_settings['valid_sample_frac']
test_sample_frac = config_settings['test_sample_frac']

epochs = config_settings['epochs']

learning_rate = config_settings['lr']
decay = config_settings['decay']

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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

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

#============================ TRAINING ===================================================================

def train(model, training_loader, validation_loader, loss_function, optimiser, logger, epochs = 5, use_attn = False):
    
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

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(training_loader.dataset)/TRAIN_BATCH_SIZE)}')

            tokens = batch["tokens"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"].to(torch.bool)

            model.zero_grad()

            tokens = tokens.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            logits = model(tokens, attention_mask, labels)

            loss = loss_function(logits, labels)
            loss.backward()

            optimiser.step()
            epoch_losses.append(np.float64(loss.cpu().detach().numpy()))
            
#============================ VALIDATION ===================================================================

            with torch.no_grad():
                model.eval()

                print(f'\tValidating {int(len(validation_loader.dataset)/VALID_BATCH_SIZE)} Batches')
                
                for index, batch in enumerate(validation_loader):

                    valid_inputs = batch["tokens"]
                    valid_targets = batch["labels"]
                    valid_attention_mask = batch["attention_mask"].to(torch.bool)

                    valid_inputs = valid_inputs.to(device)
                    valid_targets = valid_targets.to(device)
                    valid_attention_mask = valid_attention_mask.to(device)
            
                    valid_logits = model(valid_inputs, valid_attention_mask, valid_targets)

                    valid_loss = loss_function(valid_logits, labels)

                    softmax = torch.nn.Softmax(dim=0)
                    valid_predictions = softmax(valid_logits)

                    test_pred = torch.argmax(valid_predictions, dim = 1)

                    if use_attn:

                        active_accuracy = valid_attention_mask.view(-1)

                        active_logits = valid_logits.view(-1, 40)
                        flattened_predictions = torch.argmax(active_logits, axis=1)

                        valid_targets = torch.masked_select(valid_targets.view(-1), valid_attention_mask.view(-1))
                        test_pred = torch.masked_select(torch.argmax((valid_logits.view(-1, 40)), axis = 1), valid_attention_mask.view(-1))

                        test_pred = flattened_predictions

                        test_pred = torch.masked_select(flattened_predictions, active_accuracy)

                        f1_score = f1(valid_targets.cpu().detach().numpy(), test_pred.cpu().detach().numpy(), average='weighted')
                        valid_accuracy = torch.sum(torch.eq(test_pred, valid_targets)).item()/test_pred.nelement()
                        # print(f'Acc: {valid_accuracy}')

                    else:

                        f1_score = f1(valid_targets.flatten().cpu().detach().numpy(), test_pred.flatten().cpu().detach().numpy(), average='weighted')
                        valid_accuracy = torch.sum(torch.eq(test_pred, valid_targets)).item()/test_pred.nelement()

                    valid_f1_scores.append(f1_score)
                    valid_accuracies.append(valid_accuracy)
                    valid_losses.append(np.float64(valid_loss.cpu().detach().numpy()))
            
            model.train()


        average_losses.append(np.average(epoch_losses))
        average_valid_f1_scores.append(np.average(valid_f1_scores))
        average_valid_accuracies.append(np.average(valid_accuracies))
        average_valid_losses.append(np.average(valid_losses))

        if logger != '':
            logger.log({'train_loss': np.sum(epoch_losses) / len(epoch_losses),
                        'validation_f1_score': np.sum(valid_f1_scores) / len(valid_f1_scores),
                        'validation_accuracy': np.sum(valid_accuracies) / len(valid_accuracies),
                        'validation_loss': np.sum(valid_losses) / len(valid_losses)})

        if epoch % 10 == 0:
            save_model(model, epoch)

    save_model(model, epochs)

    print(f'Validation F1 Scores: {average_valid_f1_scores}')
    return average_losses, average_valid_f1_scores

#============================ TESTING ===================================================================

def test(model, testing_loader, ids_to_labels, logger):
    with torch.no_grad():

        accuracies = []
        y_true = []
        y_pred = []
        f1_scores = []

        model.eval()

        print(f'Testing {int(len(testing_loader.dataset)/VALID_BATCH_SIZE)} batches')

        for idx, batch in enumerate(testing_loader):

            inputs = batch["tokens"]
            targets = batch["labels"]
            attention_mask = batch["attention_mask"]

            inputs = inputs.to(device)
            targets = targets.to(device)
    
            logits = model(inputs, attention_mask, targets)

            softmax = torch.nn.Softmax(dim=0)
            predictions = softmax(logits)

            test_predictions = torch.argmax(predictions, dim = 1)

            f1_score = f1(targets.flatten().cpu().detach().numpy(), test_predictions.flatten().cpu().detach().numpy(), average='weighted')

            y_true.append(targets)
            y_pred.append(test_predictions)
            f1_scores.append(f1_score)

            accuracy = torch.sum(torch.eq(test_predictions, targets)).item()/test_predictions.nelement()

            accuracies.append(accuracy)
    
    if logger != '':
        logger.log({'test_accuracy': np.sum(accuracies) / len(accuracies),
                    'test_f1_score': np.sum(f1_scores) / len(f1_scores)})

    print(f'Avg f1 score: {np.sum(f1_scores) / len(f1_scores)}')

    y_true_labels = []
    y_pred_labels = []

    for labels in y_true:
        y_true_labels.append([[ids_to_labels.get(np.int64(label.cpu().item())) for label in tens_labels] for tens_labels in labels])

    for labels in y_pred:
        y_pred_labels.append([[ids_to_labels.get(np.int64(label.cpu().item())) for label in tens_labels] for tens_labels in labels])

    matrix = confusion_matrix(list(itertools.chain.from_iterable(y_true_labels[0])), list(itertools.chain.from_iterable(y_pred_labels[0])), labels=list(ids_to_labels.values()), normalize='true')

    cm = ConfusionMatrixDisplay(matrix/np.sum(matrix), display_labels=list(ids_to_labels.values()))

    fig, ax = plt.subplots(figsize=(30, 30))
    cm.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.close()

    report = classification_report(y_true_labels[0], y_pred_labels[0], output_dict = True, zero_division = 0)
    print(f'classification_report:\n{classification_report(y_true_labels[0], y_pred_labels[0], output_dict = False, zero_division = 0)}')

    plt.figure(figsize = (5, 15))
    ax = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, cmap = 'coolwarm', annot=True)
    plt.tight_layout()
    # plt.show()
    plt.close()

    if logger != '':
        logger.log({'classification_report': wandb.Image(ax.figure),
                    'confusion_matrix': wandb.Image(fig.figure)})
    
    return accuracies

#============================ LOADING DATA ===================================================================

dataset = NERSimplerDocuments()
vocab = dataset.get_vocab()
labels_to_id = dataset.get_labels_to_id()
ids_to_labels = dict(map(reversed, labels_to_id.items()))

train_data, test_data, valid_data = load_data(dataset)

training_set = CustomSimplerDataset(train_data, labels_to_id, vocab, train_sample_frac)
testing_set = CustomSimplerDataset(test_data, labels_to_id, vocab, test_sample_frac)
validation_set = CustomSimplerDataset(valid_data, labels_to_id, vocab, valid_sample_frac)

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)

print(f'Dataset len: {len(training_loader.dataset)}')

tokens_to_id = {ch:i for i,ch in enumerate(vocab)}
print(f'Unique Tokens:{len(tokens_to_id)}')

#============================ CALLING FUNCTIONS ===================================================================

logger = ''
wandb_logger = Logger(f"inm706_cw_inference_lstm_mha_attn_test", project='inm706_cw_simpler_dataset')
logger = wandb_logger.get_logger()

# model = baseline_lstm_model.LSTMModel('_Base_LSTM_sgd_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)
model = lstm_mha_attn_model.LSTMAttnModel('LSTM_MHA_sgd', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)
# model = bilstmcrf_model.BiLSTMCRFModel('BILSTMCRF_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)
# model = bilstm_model.BiLSTMModel('BILSTM_test', embedding_dim, hidden_dim, len(vocab), len(labels_to_id), labels_to_id, device)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=labels_to_id['[PAD]'])

optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = decay)
# optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = decay)

# Load checkpoint for further training
# checkpoint_dir = './checkpoints/simpler_dataset/baseline_lstm_1/saved_model_ep_10.pkl'
# model = load_model(checkpoint_dir)

# losses, valid_f1_scores = train(model, training_loader, validation_loader, loss_function, optimiser, logger, epochs, use_attn=False)

# Load checkpoint for testing
checkpoint_dir = './trained_models/LSTM_MHA.pkl'
model = load_model(checkpoint_dir)

test_accuracies = test(model, testing_loader, ids_to_labels, logger)
print(f'Test Accuracies: {test_accuracies}')