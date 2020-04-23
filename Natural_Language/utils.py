'''
@author Michael Lepori
@author Tom McCoy
@date 11/5/19

Utilities for natural language agreement task
'''

import numpy as np

# Orginally written by Tom McCoy
def create_embedding_dictionary(emb_file, f_dim, filler_to_index, index_to_filler,  unseen_words="zero"):
	embedding_dict = {}
	embed_file = open(emb_file, "r")
	for line in embed_file:
		parts = line.strip().split()
		if len(parts) == f_dim + 1:
			embedding_dict[parts[0]] = list(map(lambda x: float(x), parts[1:]))

	matrix_len = len(filler_to_index.keys())

	weights_matrix = np.zeros((matrix_len, f_dim))

	for i in range(matrix_len):
		word = index_to_filler[i]
		if word in embedding_dict:
			weights_matrix[i] = embedding_dict[word]

		elif word == "***mask***":
			weights_matrix[i] = np.ones((f_dim,))

		else:

			if unseen_words == "random":
				weights_matrix[i] = np.random.normal(scale=0.6, size=(f_dim,))
			elif unseen_words == "zero":
				pass # It was initialized as zero, so don't need to do anything
			else:
				print("Invalid choice for embeddings of unseen words")


	return weights_matrix


def create_word_idx_matrices(vocab_file, freq_threshold=1000):

    f = open(vocab_file)
    idx = 0

    word2idx = {}
    idx2word = {}

    word2idx['***mask***'] = idx
    idx2word[idx] = '***mask***'

    idx += 1

    word2idx['***unk***'] = idx
    idx2word[idx] = '***unk***'

    idx += 1
    for line in f:
        if line.startswith(' '):   # empty string token
            continue
        word, pos, count = line.strip().split()
        count = int(count)
        if len(word) > 1 and count >= freq_threshold and word not in word2idx.keys():
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

    return word2idx, idx2word
