# Representations_Of_Syntax
Code and Resources for the paper 'Representations of Syntax [MASK] Useful: Effects of Constituency and Dependency Structure in Recursive LSTMs' by Lepori, Linzen, and McCoy.

This repository is structured as follows:
- Artificial Corpus Examples: Contains code to train and run models on artificial corpora. This takes very little time, compared to the natural language training. Thus, it may be useful as a toy example. 
- Corpus Processing: Contains files of helper functions to convert parse trees into representations that the Tree LSTMs can use.
- Models: Contains the code implementing the models from Tai et al. (2015) https://arxiv.org/abs/1503.00075, a bidirectional LSTM, and the head-lexicalized tree LSTM.
- Natural Language: Contains the Code to process the LGD dataset from Linzen et al (2016) https://arxiv.org/abs/1611.01368. Also contains the scripts ran on the Maryland Advanced Research Computing Center (MARCC) to obtain results on the natural language corpora. All of these scripts take a long time to run.
- Natural and Artificial: Contains the code to test pretrained models on the artificial test set. To reproduce our results on the artificial test set, configure the main method of the script test_artificial.py, load in the correct pretrained model (either before or after augmentation), and run. Also contains the code to augment the pretrained models using a variety of PCFG-generated corpora. This also contains example scripts used to test the augmented models on natural language on the Maryland Advanced Research Computing Center (MARCC).
- Pretrained Models: Contains all models used in the final paper.
