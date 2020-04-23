import sys
sys.path.append("..")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from nltk.parse import CoreNLPParser
from nltk.parse import CoreNLPDependencyParser
import numpy as np
import nltk.treetransforms as transforms
import csv
import copy

from random import shuffle
import random

from Corpus_Processing import bracketer
from Corpus_Processing import dependency_utils
import utils
import pickle

random.seed(9)

# Set up Stanford Parsers
parser = CoreNLPParser(url='http://localhost:9000') 
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000') 

print('Processing word vectors')
word2idx, idx2word = utils.create_word_idx_matrices('./data/wiki.vocab.txt')
pickle.dump(word2idx, open('./data/word2idx.pkl', 'wb'))

glove_embeds = utils.create_embedding_dictionary('./data/glove/glove.6B.100d.txt', 100, word2idx, idx2word)
print('Done Processing Word Vectors')

pickle.dump(glove_embeds, open('./data/embed_matrix.pkl', 'wb'))

print('Processing Dataset')

# Perform preprocessing and make trees from the LGD dataset
with open('./data/full_lgd_dataset.tsv') as datafile:

    reader = csv.reader(datafile, delimiter='\t')

    progress = 0
    not_processed = 0

    sing_verb_text = []
    plur_verb_text = []

    sing_verb_const_tree = []
    plur_verb_const_tree = []

    sing_verb_dep_tree = []
    plur_verb_dep_tree = []

    sing_verb_dep_tags = []
    plur_verb_dep_tags = []

    sing_attractors = []
    plur_attractors = []

    # Batch Variables
    batch = 0
    sentences_unmasked = []
    sentences_masked = []
    split_unmasked = []

    attractors_batch = []
    answers = []

    for row in reader:

        if progress % 10000 == 0:
            print(progress)

        line_full = row[1].strip()
        line_mask = row[2].strip()

        line_mask = line_mask.replace("'", "").replace(')', '').replace('(', '').replace('-', ' ')
        line_full = line_full.replace("'", "").replace(')', '').replace('(', '').replace('-', ' ')

        answer = int(row[5])
        attractors = int(row[0])

        sentences_masked.append(line_mask)
        sentences_unmasked.append(line_full)

        split_unmasked.append(line_full.split())

        attractors_batch.append(attractors)
        answers.append(answer)

        batch += 1

        progress += 1

        # Use Stanford Parsers on batches on 100 sentences to speed up the process
        if batch == 100:
 
            try:
                stanford_trees = [next(it) for it in list(parser.parse_sents(split_unmasked))]
                stanford_parses = [next(it) for it in dep_parser.parse_sents(split_unmasked)]

            except:
                not_processed += 100
                stanford_trees =[]

            for i in range(len(stanford_trees)):

                try:
                    stanford_tree = stanford_trees[i]
                    stanford_parse = stanford_parses[i]
                    masked_sent = sentences_masked[i]
                    unmasked_sent = sentences_unmasked[i]
                    curr_attractor = attractors_batch[i]
                    curr_answer = answers[i]

                    transforms.chomsky_normal_form(stanford_tree)
                    stanford_tree = str(stanford_tree).replace('\n', '')
                    tree = bracketer.parse_to_tree_input(bracketer.convert_paren_form_to_bracket(stanford_tree))

                    dummy_tree = copy.deepcopy(tree)
                    str_const_tree = dependency_utils.string_const_tree(unmasked_sent, dummy_tree)

                    # Transform Stanford output into a form usable by the Tree LSTMs
                    govs = dependency_utils.convert_stanford_to_gov_dict(stanford_parse)
                    rank = dependency_utils.order_strings_by_dep(unmasked_sent, govs)
                    dep_tags = dependency_utils.create_dep_tags(unmasked_sent, str_const_tree, rank)
                    ordering = dependency_utils.order_doc(unmasked_sent, govs)
                    dep_tree = dependency_utils.create_dependency_tree(govs, unmasked_sent, ordering)
                    
                    if curr_answer == 1:
                        
                        sing_verb_text.append(masked_sent)
                        sing_verb_const_tree.append(tree)
                        sing_verb_dep_tags.append(dep_tags)
                        sing_verb_dep_tree.append(dep_tree)
                        sing_attractors.append(curr_attractor)

                    else:
                        plur_verb_text.append(masked_sent)
                        plur_verb_const_tree.append(tree)
                        plur_verb_dep_tags.append(dep_tags)
                        plur_verb_dep_tree.append(dep_tree)  
                        plur_attractors.append(curr_attractor)    

                except:
                        not_processed += 1

            batch = 0
            sentences_unmasked = []
            sentences_masked = []

            attractors_batch = []
            answers = []
            split_unmasked = []

    datafile.close()
    print(f'Done processing dataset: {not_processed} sentences not processed')

    # Ensure that class labels are balanced
    per_class = min([len(sing_verb_text), len(plur_verb_text)])

    print(f'{per_class} elements per class')

    sing_dataset = []

    # Places all trees and tags into a list, then forms a list of these lists
    i = 0
    for idx in range(len(sing_verb_text)):
        line = sing_verb_text[idx]
        const_tree = sing_verb_const_tree[idx]
        dep_tags = sing_verb_dep_tags[idx]
        dep_tree = sing_verb_dep_tree[idx]
        attractors = sing_attractors[idx]
        element = []
        tokens = line.split()
        indices = []
        for tok in tokens:
            if tok in word2idx.keys():
                indices.append(word2idx[tok])
            else:
                indices.append(word2idx['***unk***'])

        element.append(indices)
        element.append(const_tree)
        element.append(dep_tags)
        element.append(dep_tree)
        element.append([1])
        element.append(attractors)

        if i < per_class:
            sing_dataset.append(element)
        i += 1

    plur_dataset = []
    i = 0
    for idx in range(len(plur_verb_text)):
        line = plur_verb_text[idx]
        const_tree = plur_verb_const_tree[idx]
        dep_tags = plur_verb_dep_tags[idx]
        dep_tree = plur_verb_dep_tree[idx]
        attractors = plur_attractors[idx]
        element = []
        tokens = line.split()
        indices = []
        for tok in tokens:
            if tok in word2idx.keys():
                indices.append(word2idx[tok])
            else:
                indices.append(word2idx['***unk***'])

        element.append(indices)
        element.append(const_tree)
        element.append(dep_tags)
        element.append(dep_tree)
        element.append([0])
        element.append(attractors)

        if i < per_class:
            plur_dataset.append(element)
        i += 1


    shuffle(plur_dataset)
    shuffle(sing_dataset)

    # Should be the same number
    print(f'length of sing: {len(sing_dataset)}')
    print(f'length of plur: {len(plur_dataset)}')

    # Equal class distribution training set
    sing_train_idx = int(len(sing_dataset) * .09)
    plur_train_idx = int(len(plur_dataset) * .09)

    # Equal class distribution validation set
    sing_val_idx = int(len(sing_dataset) * .091)
    plur_val_idx = int(len(plur_dataset) * .091)

    plur_train = plur_dataset[:plur_train_idx]
    sing_train = sing_dataset[:sing_train_idx]

    plur_val = plur_dataset[plur_train_idx : plur_val_idx]
    sing_val = sing_dataset[sing_train_idx : sing_val_idx]

    plur_full_test = plur_dataset[plur_val_idx : ]
    sing_full_test = sing_dataset[sing_val_idx : ]


    train_set = sing_train + plur_train
    val_set = sing_val + plur_val

    print('Train Set Length: ' + str(len(train_set)))
    print('Val Set Length: ' + str(len(val_set)))

    full_test = sing_full_test + plur_full_test

    shuffle(train_set)
    shuffle(val_set)
    shuffle(full_test)

    any_attractors = []
    one_attractor = []
    two_attractors = []
    three_attractors = []
    four_attractors = []
    no_attractors = []

    # For each individual test set, there may not be an equal class distribution
    for element in full_test:

        if element[5] == 1:
            one_attractor.append(element)
            any_attractors.append(element)
        if element[5] == 2:
            two_attractors.append(element)
            any_attractors.append(element)
        if element[5] == 3:
            three_attractors.append(element)
            any_attractors.append(element)
        if element[5] == 4:
            four_attractors.append(element)
            any_attractors.append(element)
        if element[5] == 0 and len(no_attractors) < 50000:
            no_attractors.append(element)

    
    print('Size of Any Attractors Dataset: ' + str(len(any_attractors)))
    print('Size of One Attractors Dataset: ' + str(len(one_attractor)))
    print('Size of Two Attractors Dataset: ' + str(len(two_attractors)))
    print('Size of Three Attractors Dataset: ' + str(len(three_attractors)))
    print('Size of Four Attractors Dataset: ' + str(len(four_attractors)))
    print('Size of No Attractors Dataset: ' + str(len(no_attractors)))



    pickle.dump(train_set, open('./data/final_train.pkl', 'wb'))
    pickle.dump(val_set, open('./data/final_val.pkl', 'wb'))
    pickle.dump(no_attractors, open('./data/final_no_attractors.pkl', 'wb'))
    #pickle.dump(full_test, open('./data/final_full_test.pkl', 'wb'))
    pickle.dump(any_attractors, open('./data/final_any_attractors.pkl', 'wb'))
    pickle.dump(one_attractor, open('./data/final_one_attractor.pkl', 'wb'))
    pickle.dump(two_attractors, open('./data/final_two_attractors.pkl', 'wb'))
    pickle.dump(three_attractors, open('./data/final_three_attractors.pkl', 'wb'))
    pickle.dump(four_attractors, open('./data/final_four_attractors.pkl', 'wb'))


    print('done dumping')

datafile.close()