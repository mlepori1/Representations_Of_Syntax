'''
@author Michael Lepori
@date 10/8/19

File containing functions to handle dependency parsing tasks for Hybrid TreeLSTM Model
'''
import stanfordnlp
import numpy as np
import copy


def string_const_tree(line, ipt_form):
    ipt_form = copy.deepcopy(ipt_form)
    line = line.split()
    line = [l + str(i) for i, l in enumerate(line)]

    first = True
    for level_idx in range(len(ipt_form)):
        for node_idx in range(len(ipt_form[level_idx])):
            if first:
                ipt_form[level_idx][node_idx] = [line[ipt_form[level_idx][node_idx][0]]]
                
            elif len(ipt_form[level_idx][node_idx]) == 1:
                ipt_form[level_idx][node_idx] = ipt_form[level_idx - 1][ipt_form[level_idx][node_idx][0]]
                
            elif len(ipt_form[level_idx][node_idx]) == 2:
                ipt_form[level_idx][node_idx] = ipt_form[level_idx - 1][ipt_form[level_idx][node_idx][0]] + ipt_form[level_idx - 1][ipt_form[level_idx][node_idx][1]]

        first = False

    return ipt_form


# Function to parse a dependency grammar
def parse_dep_grammar(filename):
    grammar = {}

    f = open(filename, 'r')
    for line in f:
        if "#" in line:
            idx = line.index("#")
            line = line[:idx]
        tokens = line.split()

        if len(tokens) == 0:
            continue

        lhs = tokens[0]
        rhs = tokens[1:-1]
        head = tokens[-1]

        if lhs not in grammar.keys():
            grammar[lhs] = []

        grammar[lhs].append((rhs, head))

    return grammar


def convert_to_gold_dep(str_tree, line, dep_gram):
    
    while str_tree.replace('))', ') )') != str_tree:
        str_tree = str_tree.replace('))', ') )')
        
    parse_array = str_tree.split()
    nonterminals = [st for st in parse_array if '(' in st]
    
    nt_children = []
    chopped_parse_array = copy.deepcopy(parse_array)
    
    nt_count = 0
    for nt in nonterminals:
        start = chopped_parse_array.index(nt)
        opens = 0
        children = []
        for i in range(start + 1, len(chopped_parse_array)):
            if opens == 0 and chopped_parse_array[i] != ')':
                children.append(chopped_parse_array[i].replace('(', '').replace(')', ''))
            if '(' in chopped_parse_array[i]:
                opens += 1
            if ')' in chopped_parse_array[i]:
                opens -= 1
            if opens == -1:
                break

        nt_children.append((nt[1:], children))
        chopped_parse_array = chopped_parse_array[start:]
        
    nt_head = {}
    nt_count = 0
    for tup in nt_children:
        for candidate in dep_gram[tup[0]]:
            if candidate[0] == tup[1]:
                nt_head['(NT' + str(nt_count)] = candidate[1]
        nt_count += 1
    
    new_parse_array = []
    ct = 0
    nt_count = 0
    for st in parse_array:
        if ')' != st and ')' in st:
            new_parse_array.pop()
            new_parse_array.append(str(ct))
            ct += 1
        elif '(' in st:
            new_parse_array.append('(NT' + str(nt_count))
            nt_count += 1
        else:
            new_parse_array.append(st)
            
    governors = {}
    while ')' in new_parse_array:
        begin_idx = new_parse_array.index(')')
        idx = begin_idx
        terminals = []
        while '(' not in new_parse_array[idx]:
            if new_parse_array[idx].isdigit():
                terminals.append(int(new_parse_array[idx]))
            idx -= 1
        headword = int(new_parse_array[idx + int(nt_head[new_parse_array[idx]]) + 1])
        for t in terminals:
            if t != headword:
                governors[t] = headword
        new_parse_array = new_parse_array[:idx] + [str(headword)] + new_parse_array[begin_idx + 1:]
    governors[int(new_parse_array[0])] = -1

    return governors


# Create a dictionary that orders words based on their depth in a dependency parse
def order_strings_by_dep(line, governors):

    total = len(line.split())
    curr = 0
    targets = [-1]
    ordering = []
    while curr < total:
        ordering_array = []
        new_targets = []
        for idx, word in enumerate(line.split()):
            if governors[idx] in targets:
                curr+=1
                ordering_array.append(line.split()[idx] + str(int(idx)))
                new_targets.append(int(idx))
        ordering.append(ordering_array)
        targets = new_targets
    rank_dict = {}

    for i in range(len(ordering)):
        for word in ordering[i]:
            rank_dict[word] = i

    return rank_dict

# Tags the words based on their rank in the dependency tree
def create_dep_tags(line, str_const_tree, rank_dict):
    line = line.split()
    line = [l + str(i) for i, l in enumerate(line)]
    tags =  str_const_tree
    for level_idx in range(len(str_const_tree)):
        for node_idx in range(len(str_const_tree[level_idx])):
            best_word = ''
            best_rank = np.inf
            for word in str_const_tree[level_idx][node_idx]:
                rank = rank_dict[word]
                if rank < best_rank:
                    best_rank = rank
                    best_word = word
            tags[level_idx][node_idx] = line.index(best_word)
    return tags


# Order the words based on their governors
def order_doc(line, governors):
    
    total = len(line.split())
    ct = 0
    level = 0
    ordering = []
    for idx, element in enumerate(line.split()):
        if governors[idx] == -1:
            ordering.append([idx])
            ct+=1
            level += 1
            
    while ct < total:
        next_level = []
        for idx, element in enumerate(line.split()):
            if governors[idx] in ordering[level-1]:
                next_level.append(idx)
                ct += 1

        level += 1
        ordering.append(next_level)
    
    ordering.reverse()

    return ordering


# Creates the final dependency tree structure based on the ordering of words
def create_dependency_tree(govs, line, full_ordering):
    
    dep_tree = []
    dep_tree.append([])
    for idx in range(len(line.split())):
        dep_tree[0].append((idx, [idx]))

    gov_set = []
    next_level = []

    level = 0
    while gov_set != set([-1]):
        governors = []
        next_level = []

        for element in dep_tree[level]:
            head_idx = element[0]
            governor = govs[head_idx]
            governors.append(governor)

        gov_set = set(governors)  

        lowest_gov = []
        for curr_level in full_ordering:
            for g in list(gov_set):
                if g in curr_level:
                    lowest_gov.append(g)
                    
            if lowest_gov != []:
                break
        
        all_indices = []
        new_entries = []
        
        for curr_lowest_gov in lowest_gov:
            
            indices = list(set([i for i, g in enumerate(governors) if g == curr_lowest_gov]))

            for i in range(len(dep_tree[level])):
                if dep_tree[level][i][0] == curr_lowest_gov:
                    indices = indices + [i]
            all_indices += indices
            new_entries.append((curr_lowest_gov, indices))
            
        next_level = copy.deepcopy(dep_tree[level])
        next_level = [(el[0], [i]) for i, el in enumerate(next_level)]
        
        all_indices.sort(reverse=True)
        for i in all_indices:
            del next_level[i]
            
        next_level += new_entries
        level += 1
        dep_tree.append(next_level)

    del dep_tree[-1]
    
    return dep_tree


# Converts Stanford's parse format into the governor dictionary format used here
def convert_stanford_to_gov_dict(parse):
    conll = parse.to_conll(4)
    conll_lines = conll.split('\n')
    govs = {}

    for idx, word in enumerate(conll_lines):
        if word.split('\t') != ['']:
            gov_idx = int(word.split('\t')[2]) - 1
            govs[idx] = gov_idx
    
    return govs