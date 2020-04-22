'''
@author Tom McCoy
@author Michael Lepori
@date 10/8/19

File containing functions to handle constituency parsing tasks for Hybrid TreeLSTM Model
'''

from nltk import Tree
from functools import reduce
# Given a tree representation, get the next level of the tree representation,
# i.e. one level up the tree, where all adjacent complete constituents are 
# merged into a single constituent (whose label is an arbitrary choice from
# the labels that were merged together)
def get_next_level(words):
    skip = False

    next_level = []
    for index, word in enumerate(words):
        if skip:
            skip = False
            continue

        if index != 0 and index != len(words) - 1:
            if words[index - 1] == "[" and words[index + 1] == "]":
                next_level = next_level[:-1]
                next_level.append(word)

                skip = True
            elif words[index - 1] not in ["[", "]"] and word not in ["[", "]"]:
                next_level = next_level[:-1]
                next_level.append(word)
            else:
                next_level.append(word)
        else:
            next_level.append(word)
            
    return next_level

# Remove single-branching constituents (not necessary, but will make things
# run a little faster)
def remove_spurious(words):
    old = words[:]
    filtered = []
    skip = False
    
    
    for index, word in enumerate(words):
        if skip:
            skip = False
            continue

        if index != 0 and index != len(words) - 1:
            if words[index - 1] == "[" and words[index + 1] == "]":
                filtered = filtered[:-1]
                filtered.append(word)

                skip = True
            else:
                filtered.append(word)
        else:
            filtered.append(word)
            
    if filtered == old:
        return filtered
    else:
        return remove_spurious(filtered)
    

# Take a level of a bracketed sentence tree and turn it into one element
# in the list that is the
# representation used by the tree-RNN code
def parse_to_pairs(words):
    word_index = 0
    pairs = []
    
    for index, word in enumerate(words):
        if word not in ["[", "]"]:
            pairs.append([word_index])
            word_index += 1
            
        if index != 0 and index != len(words) - 1:
            if words[index - 1] not in ["[", "]"] and words[index] not in ["[", "]"]:
                pairs = pairs[:-2]
                pairs.append([word_index - 2, word_index - 1])
                
    return pairs


# Convert a bracketed sentence to the format used by the tree RNN
def parse_to_tree_input(bracketed_sentence):
    words = bracketed_sentence.split()
    done = False
    parse_representation = []

    while not done:
        parse_representation.append(parse_to_pairs(words))
        next_level = remove_spurious(get_next_level(words))
        if next_level == words:
            done = True

        words = next_level

    return parse_representation


def convert_paren_form_to_bracket(parse):
    parse = parse.split()
    for i in range(len(parse)):
        if parse[i][0] == '(':
            parse[i] = '['
        parse[i] = parse[i].replace(')', ' ]')
    
    return ' '.join(parse)


def get_string_from_parse(parse):
    parse = parse.split()
    for i in range(len(parse)):
        if parse[i][0] == '(':
            parse[i] = ''
        parse[i] = parse[i].replace(')', '')
    
    return ' '.join(' '.join(parse).split())
