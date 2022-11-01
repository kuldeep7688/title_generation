import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
    
import math
from tqdm import tqdm
import time

import pandas as pd
from rouge import Rouge

from utils import Vocab, prepare_data, populate_vocab, count_parameters, Dataset, generate_title
from models.model_enc_dec_with_attention import EncoderRnn, DecoderRnn, Seq2Seq


if __name__ == "__main__":
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    UNK_TOKEN = 3
    CONTENT_MAX_LENGTH = 100
    TITLE_MAX_LENGTH = 8

    train_pairs = prepare_data('../../courses/cse_842/bbc_data/train_split.csv')
    test_pairs = prepare_data('../../courses/cse_842/bbc_data/test_split.csv')

    vocab = Vocab('title_content')
    populate_vocab(vocab, train_pairs)


    EMBEDDING_DIM = 50
    HIDDEN_DIM = 64
    VOCAB_SIZE = vocab.n_words
    NUM_LAYERS_ENCODER = 2
    NUM_LAYERS_DECODER = 1

    encoder = EncoderRnn(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        enc_hidden_state_size=HIDDEN_DIM,
        dec_hidden_state_size=HIDDEN_DIM,
        num_layers=NUM_LAYERS_ENCODER,
        dropout=0.2,
        bidirectional=True
    )

    decoder = DecoderRnn(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        enc_hidden_state_size=HIDDEN_DIM,
        dec_hidden_state_size=HIDDEN_DIM,
        num_layers=NUM_LAYERS_DECODER,
        dropout=0.2,
        bidirectional=False
    )

    model = Seq2Seq(
        encoder, decoder, 0, device
    )

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    model = model.to(device)
    criterion = criterion.to(device)
    params = {
        'batch_size': 32,
        'shuffle': True,
        #   'num_workers': 6,
    }

    # Generators
    training_set = Dataset(
        train_pairs, vocab=vocab, max_len_title=TITLE_MAX_LENGTH, 
        max_len_content=CONTENT_MAX_LENGTH
    )
    training_generator = data.DataLoader(training_set, **params)

    val_set = Dataset(
        test_pairs, vocab=vocab, max_len_title=TITLE_MAX_LENGTH, 
        max_len_content=CONTENT_MAX_LENGTH
    )
    val_generator = data.DataLoader(val_set, **params)

    N_EPOCHS = 50
    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, training_generator, optimizer, criterion,
                        CLIP, params["batch_size"], device, teacher_forcing_ratio=0.1)
        valid_loss = evaluate(model, val_generator, optimizer, criterion,
                        CLIP, params["batch_size"], device, teacher_forcing_ratio=0.1)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    

    # evaluation and results on test set
    targets = []
    predictions = []
    for p in test_pairs:
        pred, tar = generate_title(p[1], p[0], vocab, model, title_max_len=TITLE_MAX_LENGTH, content_max_len=CONTENT_MAX_LENGTH)
        targets.append(tar)
        predictions.append(pred)
        print()

    rouge = Rouge()
    scores = rouge.get_scores(predictions, targets, avg=True)
    print(scores)