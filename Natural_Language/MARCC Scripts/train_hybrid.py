import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import csv
import copy

from torch import optim
from torch.nn import BCELoss
from torch.optim import Adam
import torch

from random import shuffle
import random

import models
import pickle


embed_matrix = pickle.load(open('./data/embed_matrix.pkl', 'rb'))
word2idx = pickle.load(open('./data/word2idx.pkl', 'rb'))

train_data = pickle.load(open('./data/final_train.pkl', 'rb'))
val_data = pickle.load(open('./data/final_val.pkl', 'rb'))

print("Train Length:"  + str(len(train_data)))
print("Val Length: " + str(len(val_data)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
############################## Train Model ##########################

model = models.TreeLSTMClassifier(100, 100, len(word2idx.keys()), 'hybrid', pretrained_embeddings=embed_matrix)

print("Parameters: " + str(count_parameters(model)))

num_epochs = 50
optimizer = Adam(model.parameters())
criterion = BCELoss()
epsilon = .0005

train_losses = []
val_losses = []
evaluate_every = 10000
loss_deltas = []
prev_loss = np.inf
best_loss = np.inf
best_model = ''
val_mean = np.inf


# Begin Training Procedure
for epoch in range(num_epochs):

    print('Epoch: ' + str(epoch + 1))

    train_loss = 0

    examples_trained_on = 0
    not_processed = 0

    # Iterate through training data
    for element in train_data:

        # Get training inputs
        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            loss = criterion(output, label)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            examples_trained_on += 1

        except:
            not_processed += 1


        # If we've trained on 'Evaluate Every' examples, run on validation set
        if examples_trained_on % evaluate_every == 0:

            val_not_processed = 0
            val_loss = 0

            # Iterate through validation data
            for element in val_data:

                # Get inputs to model
                seq = element[0]
                const_tree = element[1]
                dep_tags = element[2]
                dep_tree = element[3]
                label = torch.FloatTensor(element[4])

                try:
                    output = model(const_tree, dep_tree, dep_tags, seq)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    optimizer.zero_grad()

                except:
                    val_not_processed += 1

            val_avg_loss = val_loss/(len(val_data) - val_not_processed)
            val_losses.append(val_avg_loss)

            # Report validation losses
            if val_not_processed != 0:
                print('Not Processed: ' + str(val_not_processed))

            print('Avg Val Loss: ' + str(val_avg_loss))

            # Update the best model, save in case of time-out
            if val_avg_loss < best_loss:
                best_model = model.state_dict()
                best_loss = val_avg_loss
                torch.save(best_model, './models/hybrid_model')
        
            # Early stopping procedure
            loss_deltas.append(prev_loss - val_avg_loss)
            
            # Average previous 5 loss deltas
            if len(loss_deltas) > 5:
                loss_deltas.pop(0)
                val_mean = np.mean(loss_deltas)

            prev_loss = val_avg_loss

        # If val loss average is small, break
        if val_mean < epsilon:
            break

    if val_mean < epsilon:
        break

    train_avg_loss = train_loss/(len(train_data) - not_processed)
    train_losses.append(train_avg_loss)

    # Print out losses
    if not_processed != 0:
        print('Not Processed: ' + str(not_processed))

    print('Avg Train Loss: ' + str(train_avg_loss))

# Save final best model
torch.save(best_model, './models/hybrid_model')