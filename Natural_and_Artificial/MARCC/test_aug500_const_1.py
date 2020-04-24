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

any_attractors = pickle.load(open('./data/final_any_attractors.pkl', 'rb'))
one_attractor = pickle.load(open('./data/final_one_attractor.pkl', 'rb'))
two_attractors = pickle.load(open('./data/final_two_attractors.pkl', 'rb'))
three_attractors = pickle.load(open('./data/final_three_attractors.pkl', 'rb'))
four_attractors = pickle.load(open('./data/final_four_attractors.pkl', 'rb'))
no_attractors = pickle.load(open('./data/final_no_attractors.pkl', 'rb'))

model = models.TreeLSTMClassifier(100, 100, len(word2idx.keys()), 'constituency', pretrained_embeddings=embed_matrix)
model.load_state_dict(torch.load('./augmented_models/aug_const_1_500_model'))

print("No Attractors: " + str(len(no_attractors)))
print("Any Attractors: " + str(len(any_attractors)))
print("One Attractor: " + str(len(one_attractor)))
print("Two Attractors: " + str(len(two_attractors)))
print("Three Attractors: " + str(len(three_attractors)))
print("Four Attractors: " + str(len(four_attractors)))


############################ Test on No Attractors Test Set

print('Running on No Attractors Set')
correct = 0

not_processed = 0
for element in no_attractors:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])
        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if not_processed != 0:
    print('Not Processed: ' + str(not_processed))
print('Accuracy on No Attractors: ' + str(correct/(len(no_attractors) - not_processed)))

############################ Test on Attractors Test Set

print('Running on Any Attractors Test Set')
correct = 0

not_processed = 0
for element in any_attractors:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if not_processed != 0:
    print('Not Processed: ' + str(not_processed))
print('Accuracy on Any Attractors Test: ' + str(correct/(len(any_attractors) - not_processed)))


############################ Test on One Attractors Test Set

print('Running on One Attractor Test Set')
correct = 0

not_processed = 0
for element in one_attractor:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if not_processed != 0:
    print("Not Processed: " + str(not_processed))
print('Accuracy on One Attractor Test: ' + str(correct/(len(one_attractor) - not_processed)))


############################ Test on Two Attractors Test Set

print('Running on Two Attractors Test Set')
correct = 0
not_processed = 0

for element in two_attractors:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if len(two_attractors) != 0:

    if not_processed != 0:
        print("Not Processed: " + str(not_processed))
    print('Accuracy on Two Attractors Test: ' + str(correct/(len(two_attractors) - not_processed)))

############################ Test on Three Attractors Test Set

print('Running on Three Attractors Test Set')
correct = 0
not_processed = 0

for element in three_attractors:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if len(three_attractors) != 0:

    if not_processed != 0:
        print("Not Processed: " + str(not_processed))
    print('Accuracy on Three Attractors Test: ' + str(correct/(len(three_attractors) - not_processed)))


############################ Test on Four Attractors Test Set

print('Running on Four Attractors Test Set')
correct = 0
not_processed = 0

for element in four_attractors:

        seq = element[0]
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            output = model(const_tree, dep_tree, dep_tags, seq)
            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

if len(four_attractors) != 0:
    if not_processed != 0:
        print("Not Processed: " + str(not_processed))
    print('Accuracy on Four Attractors Test: ' + str(correct/(len(four_attractors) - not_processed)))