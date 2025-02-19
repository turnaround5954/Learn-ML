#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        # YOUR CODE HERE for part 1h
        char_embed_size = 50
        self.vocab = vocab
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.emb_layer = nn.Embedding(len(vocab.char2id),
                                      char_embed_size,
                                      padding_idx=vocab.char_pad)
        self.cnn_layer = CNN(char_embed_size, word_embed_size)
        self.highway_layer = Highway(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)

        # END YOUR CODE

    def forward(self, inputs):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # YOUR CODE HERE for part 1h
        sent_len, batch_size, word_len = inputs.size()
        x_emb = self.emb_layer(inputs)
        x_reshape = x_emb.view(sent_len * batch_size,
                               word_len,
                               self.char_embed_size).permute(0, 2, 1).contiguous()
        x_conv_out = self.cnn_layer(x_reshape)
        x_highway = self.highway_layer(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb.view(sent_len, batch_size, self.word_embed_size).contiguous()

        # END YOUR CODE
