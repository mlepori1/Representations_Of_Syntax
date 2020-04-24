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
import pickle


embed_matrix = pickle.load(open('./data/embed_matrix.pkl', 'rb'))
word2idx = pickle.load(open('./data/word2idx.pkl', 'rb'))


import stanfordnlp

random.seed(9)

#All words seen during training
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
                "nice", "happy", "sad", "dangerous", "soggy", "sloppy", "***mask***"]


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
            print(str(round(len(plur_verb_text)/per_class, 3)), "% : plurals")
            print(str(round(len(sing_verb_text)/per_class, 3)), "% : singulars")
            print(progress, " lines processed")


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
                tokens[tok_idx] = "***mask***"
        indices = []
        for tok in tokens:
            if tok not in word2idx.keys():
                print(tok)
                indices.append(word2idx['***unk***'])
            else:
                indices.append(word2idx[tok])
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
                tokens[tok_idx] = "***mask***"
        indices = []
        for tok in tokens:
            if tok not in word2idx.keys():
                indices.append(word2idx['***unk***'])
            else:
                indices.append(word2idx[tok])        
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
        print("first ", str(len(train_set)))
        print("second ", str(len(test_set)))
        return train_set, test_set
    else:
        print("first ", str(len(train_set)))
        return train_set


# Script to load in and augment a model
if __name__ == "__main__":

    dep_gram = dependency_utils.parse_dep_grammar('./grammars/agreement_over_tree_grammar_deps.txt')
    train_set, test_set = make_set('./corpora/agreement_corpus.txt', 350, 100, True, dep_gram)
    aug_set = make_set('./corpora/gen_corpus.txt', 250, 250, False, dep_gram)
    
    ############################## Train Model #####################

    # Select whether you would like to augment the BiLSTM or a tree model

    model = models.BidirectionalLSTM(100, 100, len(word2idx.keys()), pretrained_embeddings=embed_matrix)
    #model = models.TreeLSTMClassifier(100, 100, len(word2idx.keys()), 'hybrid', pretrained_embeddings=embed_matrix)

    # Load in the model pretrained on natural language
    model.load_state_dict(torch.load('./augmented_models/biLSTM_model_1'))

    
    optimizer = Adam(model.parameters())
    criterion = BCELoss()

    train_loss = 0

    for element in aug_set:

        seq = element[0]
        mask_idx = seq.index(word2idx['***mask***'])
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])
        #output = model(const_tree, dep_tree, dep_tags, seq) # For Tree Models
        output = model(seq)[mask_idx]  # For BiLSTM
        loss = criterion(output, label)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #torch.save(model.state_dict(), './augmented_models/aug_biLSTM_3_500_model')
    
    ###### Extract 400 Sentence Test Set, Disjoint From the Augmentation Set ########

    test_plurs = []
    test_sings = []

    for element1 in test_set:
        if element1 not in aug_set:
            if element1[4][0] == 1 and len(test_plurs) < 200 and element1 not in test_plurs:
                test_plurs.append(element1)

            if element1[4][0] == 0 and len(test_sings) < 200 and element1 not in test_sings:
                test_sings.append(element1)

    non_overlapped_test = test_sings + test_plurs
    print(f'Artificial Test Set Length: {len(non_overlapped_test)}')
    shuffle(non_overlapped_test)

    ############################ Test on Artificial Test Set  ##############

    print('Running on Test Set')
    correct = 0
    not_processed = 0
    processed = 0
    for element in non_overlapped_test:

        processed += 1
        seq = element[0]
        mask_idx = seq.index(word2idx['***mask***'])
        const_tree = element[1]
        dep_tags = element[2]
        dep_tree = element[3]
        label = torch.FloatTensor(element[4])

        try:
            #output = model(const_tree, dep_tree, dep_tags, seq)    # For Tree Models
            output = model(seq)[mask_idx]   # For BiLSTM

            if (output > .5 and label == 1) or (output < .5 and label == 0):
                correct += 1
        except:
            not_processed += 1

    print('Accuracy on Artificial Test: ' + str(correct/(len(non_overlapped_test) - not_processed)))