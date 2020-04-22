'''
This file allows users to train and test all models on the artificial corpora used in
'Representations of Syntax [MASK] Useful:Effects of Constituency and 
Dependency Structure in Recursive LSTMs' by Lepori, McCoy, and Linzen.
'''

import sys
import traceback
sys.path.append("..")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import copy 

from Models import models
from Corpus_Processing import bracketer
from torch import optim
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from random import shuffle
import random
from Corpus_Processing import dependency_utils
import numpy as np

import stanfordnlp

random.seed(9)

# All words seen during training when using the artificial corpora
dictionary = ["eats", "pleases", "loves", "likes", "hates", "destroys",
                "creates", "fights", "bites", "shoots", "arrests", "takes",
                "leaves", "buys", "brings", "carries", "kicks",
                "eat", "please", "love", "like", "hate", "destroy",
                "create", "fight", "bite", "shoot", "arrest", "take",
                "leave", "buy", "bring", "carry", "kick", "planes",
        	    "bakers", "jesters", "birds", "cars", "dancers", "singers",
                "presidents", "squirrels", "squids", "clouds", "actors", "doctors",
                "nurses", "chairs", "students", "teachers", "ferns", "the",
                "plane", "baker", "jester", "bird", "car", "dancer", "singer",
                "president", "squirrel", "squid", "cloud", "actor", "doctor",
                "nurse", "chair", "student", "teacher", "fern", "on", "by", "near", "around",
                "fancy", "green", "handsome", "pretty", "large", "big", "scary", "spooky", 
                "nice", "happy", "sad", "dangerous", "soggy", "sloppy", "MASK"]



def make_set(filename, per_class, first_ct, second, dep_grammar):
    '''
    This function takes in a corpus file and it's dependency grammar file, and generates
    training (and test) sets. 
    @arg filename: The name of the corpus file
    @arg per_class: The number of examples of each class (plural/singular)
    that are to be extracted from the corpus. 
    @arg first_ct: The number of examples of each class to be put into the first dataset generated 
    from the corpus
    @arg second: A boolean determining whether there will be two datasets generated from this corpus. 
    If True, then any examples that are not put in the first corpus will be placed into the second corpus.
    Thus, the total number of examples in this second corpus is (2*per_class - 2*first_ct)
    @arg dep_grammar: A file describing the dependency grammar of the corpus
    '''
    file = open(filename, 'r')

    sing_verb_text = []
    plur_verb_text = []

    sing_verb_const_tree = []
    plur_verb_const_tree = []

    sing_verb_dep_tree = []
    plur_verb_dep_tree = []

    sing_verb_dep_tags = []
    plur_verb_dep_tags = []

    sing_verbs = ["eats", "pleases", "loves", "likes", "hates", "destroys",
                "creates", "fights", "bites", "shoots", "arrests", "takes",
                "leaves", "buys", "brings", "carries", "kicks"]

    progress = 0
    for line in file:

        progress += 1
        if progress % 100 == 0:
            progress = progress

        if len(sing_verb_text) >= per_class and len(plur_verb_text) >= per_class:
            break


        tree = bracketer.parse_to_tree_input(bracketer.convert_paren_form_to_bracket(line))
        str_const_parse = line
        line = bracketer.get_string_from_parse(line)

        s1 = set(sing_verbs)
        s2 = set(line.split())

        # Ensures that equal numbers of singular and plural examples are extracted
        if s1.intersection(s2):
            if line not in sing_verb_text and len(sing_verb_text) < per_class:

                # Generates the dependency tree using a series of helper functions
                dummy_tree = copy.deepcopy(tree)
                str_const_tree = dependency_utils.string_const_tree(line, dummy_tree)
                govs = dependency_utils.convert_to_gold_dep(str_const_parse, line, dep_grammar)
                rank = dependency_utils.order_strings_by_dep(line, govs)
                dep_tags = dependency_utils.create_dep_tags(line, str_const_tree, rank)
                ordering = dependency_utils.order_doc(line, govs)
                dep_tree = dependency_utils.create_dependency_tree(govs, line, ordering)

                sing_verb_text.append(line)
                sing_verb_const_tree.append(tree)
                sing_verb_dep_tags.append(dep_tags)
                sing_verb_dep_tree.append(dep_tree)
        else:
            if line not in plur_verb_text and len(plur_verb_text) < per_class:

                # Generates the dependency tree using a series of helper functions
                dummy_tree = copy.deepcopy(tree)
                str_const_tree = dependency_utils.string_const_tree(line, dummy_tree)
                govs = dependency_utils.convert_to_gold_dep(str_const_parse, line, dep_grammar)
                rank = dependency_utils.order_strings_by_dep(line, govs)
                dep_tags = dependency_utils.create_dep_tags(line, str_const_tree, rank)
                ordering = dependency_utils.order_doc(line, govs)
                dep_tree = dependency_utils.create_dependency_tree(govs, line, ordering)

                plur_verb_text.append(line)
                plur_verb_const_tree.append(tree)
                plur_verb_dep_tags.append(dep_tags)
                plur_verb_dep_tree.append(dep_tree)



    print(filename)
    print(f'Number of singular verb sentences: {len(sing_verb_text)}')
    print(f'Number of plural verb sentences: {len(plur_verb_text)}')

    file.close()

    # Now that the examples are extracted, create a simple dataset containing all of the relevant parses,
    # Also, preprocess the sentences, masking out the main verb and replacing the token strings with indices 

    all_verbs = ["eats", "pleases", "loves", "likes", "hates", "destroys",
                "creates", "fights", "bites", "shoots", "arrests", "takes",
                "leaves", "buys", "brings", "carries", "kicks",
                "eat", "please", "love", "like", "hate", "destroy",
                "create", "fight", "bite", "shoot", "arrest", "take",
                "leave", "buy", "bring", "carry", "kick"]

    train_set = []
    test_set = []

    i = 0
    for idx in range(len(sing_verb_text)):
        line = sing_verb_text[idx]
        const_tree = sing_verb_const_tree[idx]
        dep_tags = sing_verb_dep_tags[idx]
        dep_tree = sing_verb_dep_tree[idx]
        element = []
        tokens = line.split()

        # Mask out the main verb in the sentence
        for tok_idx in range(len(tokens)):
            if tokens[tok_idx] in all_verbs:
                tokens[tok_idx] = "MASK"
        indices = []
        for tok in tokens:
            indices.append(dictionary.index(tok))

        element.append(indices)
        element.append(const_tree)
        element.append(dep_tags)
        element.append(dep_tree)
        element.append([0])

        # Split the examples between the first and second datasets
        if i < first_ct:
            train_set.append(element)
        else:
            test_set.append(element)

        i += 1

    i = 0 
    for idx in range(len(plur_verb_text)):
        line = plur_verb_text[idx]
        const_tree = plur_verb_const_tree[idx]
        dep_tags = plur_verb_dep_tags[idx]
        dep_tree = plur_verb_dep_tree[idx]
        element = []
        tokens = line.split()

        # Mask out the main verb in the sentence

        for tok_idx in range(len(tokens)):
            if tokens[tok_idx] in all_verbs:
                tokens[tok_idx] = "MASK"
        indices = []
        for tok in tokens:
            indices.append(dictionary.index(tok))
        
        element.append(indices)
        element.append(const_tree)
        element.append(dep_tags)
        element.append(dep_tree)
        element.append([1])

        # Split the examples between the first and second datasets
        if i < first_ct:
            train_set.append(element)
        else:
            test_set.append(element)
        i += 1

    shuffle(train_set)
    shuffle(test_set)

    if second:
        return train_set, test_set
    else:
        return train_set



# Script to carry out testing and training
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Please specify which model you would like to use")
        exit()

    model_type = sys.argv[1]
    model_types = ["biLSTM", "constituency", "dependency", "hybrid"]
    
    if model_type not in model_types:
        print("Invalid model selection\nAllowed models: biLSTM, constituency, dependency, hybrid")
        exit()

    print('Gold Constituency and Gold Dependency Parses Experiment')

    # Generate datasets using the function above
    dep_gram = dependency_utils.parse_dep_grammar('./grammars/agreement_over_tree_grammar_deps.txt')
    train_set, test_set = make_set('./corpora/agreement_corpus.txt', 300, 200, True, dep_gram)
    gen_set = make_set('./corpora/agreement_generalization_corpus.txt', 500, 500, False, dep_gram)

    ############################## Train Model ################################

    if model_type !='biLSTM':
        model = models.TreeLSTMClassifier(64, 64, len(dictionary), model_type)
    else:
        model = models.BidirectionalLSTM(64, 64, len(dictionary))

    num_epochs = 50
    optimizer = Adam(model.parameters())
    criterion = BCELoss()
    epsilon = .0005

    train_losses = []

    prev_loss = np.inf

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch}')
        train_loss = 0

        for element in train_set:

            seq = element[0]
            mask_idx = seq.index(dictionary.index('MASK'))
            const_tree = element[1]
            dep_tags = element[2]
            dep_tree = element[3]
            label = torch.FloatTensor(element[4])

            if model_type != "biLSTM":
                output = model(const_tree, dep_tree, dep_tags, seq)
            else:
                output = model(seq)[mask_idx]

            loss = criterion(output, label)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        train_losses.append(train_loss)

        print(f'LOSS: {train_loss.item()/len(train_set)}')

        # Early stopping procedure
        if prev_loss - (train_loss.item()/len(train_set)) < epsilon and epoch >= 4:
            break

        prev_loss = (train_loss.item()/len(train_set))

    ############################ Test on Test Set ##################
    correct = 0

    for element in test_set:
            seq = element[0]
            mask_idx = seq.index(dictionary.index('MASK'))
            const_tree = element[1]
            dep_tags = element[2]
            dep_tree = element[3]
            label = torch.FloatTensor(element[4])

            if model_type != "biLSTM":
                output = model(const_tree, dep_tree, dep_tags, seq)
            else:
                output = model(seq)[mask_idx]

            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1

    print(f'Accuracy: {correct/(len(test_set) + .0)}')


    ############################ Test on Generalization Set #################

    correct = 0

    for element in gen_set:
            seq = element[0]
            mask_idx = seq.index(dictionary.index('MASK'))
            const_tree = element[1]
            dep_tags = element[2]
            dep_tree = element[3]
            label = torch.FloatTensor(element[4])

            if model_type != "biLSTM":
                output = model(const_tree, dep_tree, dep_tags, seq)
            else:
                output = model(seq)[mask_idx]

            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1

    print(f'Accuracy on Generalization Set: {correct/(len(gen_set) + .0)}')