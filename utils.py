import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
    
import torch
from torch.utils import data

import math
from tqdm import tqdm
import time

import pandas as pd
from rouge import Rouge



class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2, 'UNK':3}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: 'UNK'}
        self.n_words = 3
        self.word2count = {}
    
    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index.keys():
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def to_json(self, file_path):
        pass
    
    def read_from_json(self, file_path):
        pass


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepare_data(
    file_path, context_max_length=CONTENT_MAX_LENGTH, 
    title_max_length=TITLE_MAX_LENGTH
):
    df = pd.read_csv(file_path)
    pairs = []
    for _, row in df.iterrows():
        pairs.append(
            [
                row['title'], row['content']
            ]
        )
    print("{} titles and content read.".format(len(pairs)))
    pairs = [[normalize_string(p[0]), normalize_string(p[1])] for p in pairs]

    return pairs


def populate_vocab(vocab, pairs):
    for ti, co in pairs:
        vocab.add_sentence(co)
    return



class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pairs, vocab, max_len_title, max_len_content):
        'Initialization'
        self.pairs = pairs
        self.max_len_title = max_len_title
        self.max_len_content = max_len_content
        self.vocab = vocab
        self.input_content = [tensorFromSentence(self.vocab, inp[1], self.max_len_content) for inp in self.pairs]
        self.output_title = [tensorFromSentence(self.vocab, inp[0], self.max_len_title) for inp in self.pairs]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pairs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = self.input_content[index]
        y = self.output_title[index]

        return X, y

def indexesFromSentence(vocab, sentence, max_len):
    l = [vocab.word2index.get(word, vocab.word2index['UNK']) for word in sentence.split()]
    if len(l) > max_len - 2:
        l = l[:max_len-2]
    
    l = [0] + l + [1] 
    if len(l) < max_len:
        for i in range(len(l), max_len):
            l.append(vocab.word2index["PAD"])
    
    return l


def tensorFromSentence(vocab, sentence, max_len):
    indexes = indexesFromSentence(vocab, sentence, max_len)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip, batch_size, device, teacher_forcing_ratio=0.25):
    model.train()
    
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator)):
        
        src = batch[0].permute(1,0,2).squeeze(-1).to(device).contiguous()
        trg = batch[1].permute(1,0,2).squeeze(-1).to(device).contiguous()
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, optimizer, criterion, clip, batch_size, device, teacher_forcing_ratio=0.2):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator)):
            src = batch[0].permute(1,0,2).squeeze(-1).to(device).contiguous()
            trg = batch[1].permute(1,0,2).squeeze(-1).to(device).contiguous()

            output = model(src, trg, teacher_forcing_ratio) #turn off teacher forcing
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def strip_eos_sos(ids, vocab):
    tokens = [vocab.index2word[w] for w in ids]
    real_tokens = []
    for tok in tokens[1:]:
        if tok == 'EOS':
            break
        real_tokens.append(tok)
    
    return ' '.join(real_tokens)


def generate_title(
    content, title, vocab, model, title_max_len, content_max_len,
    print_outputs=False
):
    model.eval()
    src = tensorFromSentence(vocab, content, content_max_len)
    trg = tensorFromSentence(vocab, title, title_max_len)
    
    out = model(src, trg, teacher_forcing_ratio=0.5)
    out = F.softmax(out, dim=2)
    predictions = out.max(2)[1].view(-1)
    pred_sentence = strip_eos_sos(predictions.tolist(), vocab)
    gt_sentence = strip_eos_sos(trg.view(-1).tolist(), vocab)
    if print_outputs:
        print("       Input == {}".format(" ".join([vocab.index2word[w] for w in src.view(-1).tolist()])))
        print("Model Output == {}".format(pred_sentence))
        print("Ground Truth == {}".format(gt_sentence))
    return pred_sentence, gt_sentence