'''
@author: Michael Lepori
@date: 7/21/19

Implementation of Tai et al's two main tree LSTM models.
The equations underlying these models are found in
"Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"

The weight matrices of the dependency model are made larger in order to make the number
of free parameters more similar to that of the other models.

Implementation of Hybrid Tree LSTM as well

Also, implementation of Bidirectional Tree LSTM
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class DependencyTreeLSTM(nn.Module):
    # Dependency Tree LSTMs recieve an input x for every node,
    # corresponding to the headword of the child nodes.
    def __init__(self, embed_size, hidden_size, vocab_size, pretrained_embeddings):
        super(DependencyTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if pretrained_embeddings is not None:
                self.embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings)})
                self.embedding.weight.requires_grad = False  

        self.w_i = nn.Linear(self.embed_size, self.hidden_size)
        self.u_i = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_f = nn.Linear(self.embed_size, self.hidden_size)
        self.u_f = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_o = nn.Linear(self.embed_size, self.hidden_size)
        self.u_o = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_u = nn.Linear(self.embed_size, self.hidden_size)
        self.u_u = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, dependency_tree, input_seq):

        embedded_seq = []

        for elt in input_seq:
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt]))).unsqueeze(0))

        current_level_c = []
        current_level_h = []

        first = True
        for level in dependency_tree:
            next_level_c = []
            next_level_h = []

            for node in level:

                if len(node[1]) == 1 and first == True:
                    x_j = embedded_seq[node[0]]
                    h_tilde_j = Variable(torch.zeros(self.hidden_size))
                    sum_term_cj = Variable(torch.zeros(self.hidden_size))

                else:
                    x_j = embedded_seq[node[0]]
                    h_tilde_j = Variable(torch.zeros(self.hidden_size))
                    sum_term_cj = Variable(torch.zeros(self.hidden_size))


                    for child in node[1]:
                        h_k = current_level_h[child]
                        c_k = current_level_c[child]
                        f_jk = torch.sigmoid(self.w_f(x_j).add(self.u_f(h_k)))
                        h_tilde_j = h_tilde_j.add(h_k)
                        sum_term_cj = sum_term_cj.add(torch.mul(f_jk, c_k))

                
                sum_term_cj = sum_term_cj.reshape(-1)
                h_tilde_j = h_tilde_j.reshape(-1)
                i_j = torch.sigmoid(self.w_i(x_j).add(self.u_i(h_tilde_j))).reshape(-1)
                o_j = torch.sigmoid(self.w_o(x_j).add(self.u_o(h_tilde_j))).reshape(-1)
                u_j = torch.tanh(self.w_u(x_j).add(self.u_u(h_tilde_j))).reshape(-1)

                c_j = torch.mul(i_j, u_j).add(sum_term_cj)
                h_j = torch.mul(o_j, torch.tanh(c_j))

                next_level_c.append(c_j)
                next_level_h.append(h_j)

            first = False

            current_level_c = next_level_c
            current_level_h = next_level_h
        
        return current_level_h, current_level_c 


class ConstituencyTreeLSTM(nn.Module):
    # Constituency Tree LSTM nodes only recieve an input
    # word x if they are leaf nodes.
    def __init__(self, embed_size, hidden_size, vocab_size, pretrained_embeddings):
        super(ConstituencyTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if pretrained_embeddings is not None:
                self.embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings)})
                self.embedding.weight.requires_grad = False  

        self.w_i = nn.Linear(self.embed_size, self.hidden_size)
        self.u_i_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_i_r = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_f = nn.Linear(self.embed_size, self.hidden_size)
        self.u_f_ll = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_lr = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_rl = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_rr = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_o = nn.Linear(self.embed_size, self.hidden_size)
        self.u_o_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_o_r = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_u = nn.Linear(self.embed_size, self.hidden_size)
        self.u_u_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_u_r = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, constituency_tree, input_seq):

        embedded_seq = []

        for elt in input_seq:
            #embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).to(device='cuda')).unsqueeze(0))
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt]))).unsqueeze(0))

        current_level_c = []
        current_level_h = []

        first_level = True

        for level in constituency_tree:

            next_level_c = []
            next_level_h = []

            for node in level:
                if first_level:              
                    x = embedded_seq[node[0]]
                    h_left = Variable(torch.zeros(self.hidden_size))
                    h_right = Variable(torch.zeros(self.hidden_size))
                    c_left = Variable(torch.zeros(self.hidden_size))
                    c_right = Variable(torch.zeros(self.hidden_size))

                elif len(node) == 1:
                    #h_left = Variable(torch.zeros(self.hidden_size).to(device='cuda'))
                    x = embedded_seq[node[0]]
                    h_left = current_level_h[node[0]]
                    h_right = Variable(torch.zeros(self.hidden_size))
                    c_left = current_level_c[node[0]]
                    c_right = Variable(torch.zeros(self.hidden_size))

                else:
                    x = Variable(torch.zeros(self.embed_size))
                    c_left = current_level_c[node[0]]
                    c_right = current_level_c[node[1]]
                    h_left = current_level_h[node[0]]
                    h_right = current_level_h[node[1]]

                sum_term_c = Variable(torch.zeros(self.hidden_size))

                i = torch.sigmoid(self.w_i(x).add(self.u_i_l(h_left)).add(self.u_i_r(h_right))).reshape(-1)

                f_l = torch.sigmoid(self.w_f(x).add(self.u_f_ll(h_left)).add(self.u_f_lr(h_right))).reshape(-1)
                f_r = torch.sigmoid(self.w_f(x).add(self.u_f_rl(h_right)).add(self.u_f_rr(h_left))).reshape(-1)

                o = torch.sigmoid(self.w_o(x).add(self.u_o_l(h_left)).add(self.u_o_r(h_right))).reshape(-1)

                u = torch.tanh(self.w_u(x).add(self.u_u_l(h_left)).add(self.u_u_r(h_right))).reshape(-1) 

                sum_term_c = sum_term_c.add(torch.mul(f_l, c_left)).add(torch.mul(f_r, c_right))   
                    
                c = torch.mul(i, u).add(sum_term_c)
                h = torch.mul(o, torch.tanh(c))

                next_level_c.append(c)
                next_level_h.append(h)

            first_level = False
            current_level_c = next_level_c
            current_level_h = next_level_h

        return current_level_h, current_level_c        


class HybridTreeLSTM(nn.Module):
    # Hybrid Tree LSTM nodes recieves an input
    # word x at all nodes.
    def __init__(self, embed_size, hidden_size, vocab_size, pretrained_embeddings):
        super(HybridTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if pretrained_embeddings is not None:
                self.embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings)})
                self.embedding.weight.requires_grad = False  

        self.w_i = nn.Linear(self.embed_size, self.hidden_size)
        self.u_i_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_i_r = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_f = nn.Linear(self.embed_size, self.hidden_size)
        self.u_f_ll = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_lr = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_rl = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_f_rr = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_o = nn.Linear(self.embed_size, self.hidden_size)
        self.u_o_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_o_r = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_u = nn.Linear(self.embed_size, self.hidden_size)
        self.u_u_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_u_r = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, constituency_tree, dependency_tags, input_seq):

        embedded_seq = []

        for elt in input_seq:
            #embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).to(device='cuda')).unsqueeze(0))
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt]))).unsqueeze(0))

        current_level_c = []
        current_level_h = []

        first_level = True

        for level_idx in range(len(constituency_tree)):

            level = constituency_tree[level_idx]

            next_level_c = []
            next_level_h = []

            for node_idx in range(len(level)):

                node = level[node_idx]

                if first_level:
                    x = embedded_seq[node[0]]
                    h_left = Variable(torch.zeros(self.hidden_size))
                    h_right = Variable(torch.zeros(self.hidden_size))
                    c_left = Variable(torch.zeros(self.hidden_size))
                    c_right = Variable(torch.zeros(self.hidden_size))

                elif len(node) == 1:
                    #h_left = Variable(torch.zeros(self.hidden_size).to(device='cuda'))
                    x = embedded_seq[dependency_tags[level_idx - 1][node_idx]]
                    c_left = current_level_c[node[0]]
                    c_right = Variable(torch.zeros(self.hidden_size))
                    h_left = current_level_h[node[0]]
                    h_right = Variable(torch.zeros(self.hidden_size))

                else:
                    x = embedded_seq[dependency_tags[level_idx - 1][node_idx]]
                    c_left = current_level_c[node[0]]
                    c_right = current_level_c[node[1]]
                    h_left = current_level_h[node[0]]
                    h_right = current_level_h[node[1]]

                sum_term_c = Variable(torch.zeros(self.hidden_size))

                i = torch.sigmoid(self.w_i(x).add(self.u_i_l(h_left)).add(self.u_i_r(h_right))).reshape(-1)

                f_l = torch.sigmoid(self.w_f(x).add(self.u_f_ll(h_left)).add(self.u_f_lr(h_right))).reshape(-1)
                f_r = torch.sigmoid(self.w_f(x).add(self.u_f_rl(h_right)).add(self.u_f_rr(h_left))).reshape(-1)

                o = torch.sigmoid(self.w_o(x).add(self.u_o_l(h_left)).add(self.u_o_r(h_right))).reshape(-1)

                u = torch.tanh(self.w_u(x).add(self.u_u_l(h_left)).add(self.u_u_r(h_right))).reshape(-1) 

                sum_term_c = sum_term_c.add(torch.mul(f_l, c_left)).add(torch.mul(f_r, c_right))   
                    
                c = torch.mul(i, u).add(sum_term_c)
                h = torch.mul(o, torch.tanh(c))

                next_level_c.append(c)
                next_level_h.append(h)

            first_level = False
            current_level_c = next_level_c
            current_level_h = next_level_h

        return current_level_h, current_level_c    

class TreeLSTMClassifier(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, tree, pretrained_embeddings=None):
        super(TreeLSTMClassifier, self).__init__()
        self.tree = tree
        self.linear = nn.Linear(hidden_size, 1)

        if self.tree == "constituency":
            self.tree_lstm = ConstituencyTreeLSTM(embed_size, hidden_size, vocab_size, pretrained_embeddings)
        elif self.tree == "hybrid":
            self.tree_lstm = HybridTreeLSTM(embed_size, hidden_size, vocab_size, pretrained_embeddings)
        elif self.tree == "dependency":
            self.tree_lstm = DependencyTreeLSTM(embed_size, hidden_size, vocab_size, pretrained_embeddings)


    def forward(self, constituency_tree, dependency_tree, dependency_tags, input_seq):
        if self.tree == "constituency":
            return torch.sigmoid(self.linear(torch.FloatTensor(self.tree_lstm(constituency_tree, input_seq)[0][0])))

        if self.tree == "hybrid":
            return torch.sigmoid(self.linear(torch.FloatTensor(self.tree_lstm(constituency_tree, dependency_tags, input_seq)[0][0])))

        if self.tree == "dependency":
            return torch.sigmoid(self.linear(torch.FloatTensor(self.tree_lstm(dependency_tree, input_seq)[0][0])))


# Based on code from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
# and https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967/2

class BidirectionalLSTM(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, pretrained_embeddings=None):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if pretrained_embeddings is not None:
                self.embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings)})
                self.embedding.weight.requires_grad = False  

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2, 1)

        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(2, 1, self.hidden_size)),   
            torch.autograd.Variable(torch.zeros(2, 1, self.hidden_size)))


    def forward(self, input_seq):

        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        embedded_seq = []
        for elt in input_seq:
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt]))).unsqueeze(0))
        embedded_seq = torch.cat(embedded_seq)


        lstm_out, self.hidden = self.lstm(embedded_seq.view(len(input_seq), 1, -1), self.hidden)

        return torch.sigmoid(self.linear(lstm_out.view(len(input_seq), -1)))