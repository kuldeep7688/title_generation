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


class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_state_size, dec_hidden_state_size, num_layers, dropout=0.5,
                bidirectional=True):
        super(EncoderRnn, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_hidden_state_size = enc_hidden_state_size
        self.dec_hidden_state_size = dec_hidden_state_size
        self.num_layers=num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.enc_hidden_state_size,
                          num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.combined_context_layer = nn.Linear(self.enc_hidden_state_size * 2, self.dec_hidden_state_size)
        
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, inp):        
        # inp = [sent_length, batch_size]

        embedded = self.dropout_layer(self.embedding(inp))
        # embedded = [sent_length, batch_size, embedding_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [seq_len, batch, num_directions * hidden_size]
        # hidden = [num_layers*num_directions, batch, hidden_size]

        combined_context = torch.tanh(self.combined_context_layer(torch.cat((hidden[-2, : ,:], hidden[-1, :, :]), dim=1)))
        # combined_context = [batch_size, dec_hidden_state_size]
        
        return outputs, combined_context


class AttentionLayer(nn.Module):
    def __init__(self, dec_hidden_state_size, enc_hidden_state_size):
        super(AttentionLayer, self).__init__()
        self.dec_hidden_state_size = dec_hidden_state_size
        self.enc_hidden_state_size = enc_hidden_state_size
        
        self.attn = nn.Linear((2 * self.enc_hidden_state_size) + self.dec_hidden_state_size, self.dec_hidden_state_size)
        
        self.v = nn.Parameter(torch.rand(self.dec_hidden_state_size))
        
    def forward(self, hidden, enc_outputs):
        # hidden = [batch_size, dec_hidden_state_size]
        # enc_outputs = [src_sent_len, batch_size, enc_hidden_state_size*2]
        
        batch_size = hidden.shape[0]
        src_seq_len = enc_outputs.shape[0]
        
        # calculating the energy 
        hidden = hidden.unsqueeze(1).repeat(1, src_seq_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # hidden = [batch_size, src_seq_len, dec_hidden_state_size]
        # enc_outputs = [batch_size, src_seq_len, 2*enc_hidden_state_size]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        # energy = [batch_size, src_seq_len, dec_hidden_state_size]
        # v = [dec_hidden_state_size]
        
        energy = energy.permute(0, 2, 1)
        # energy = [batch_size, dec_hidden_state_size, src_seq_len]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        attn = torch.bmm(v, energy)
        # attn = [batch_size, 1, src_seq_len]
        
        attn = attn.squeeze(1)
        # attn = [batch_size, src_seq_len]

        return F.softmax(attn, dim=1)


class DecoderRnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_state_size, 
                 dec_hidden_state_size, num_layers=1, dropout=0.5, 
                 bidirectional=False):
        
        super(DecoderRnn, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_hidden_state_size = enc_hidden_state_size
        self.dec_hidden_state_size = dec_hidden_state_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.attn_layer = AttentionLayer(self.dec_hidden_state_size, self.enc_hidden_state_size)
        
        self.rnn = nn.GRU(input_size=(2 * self.enc_hidden_state_size) + self.embedding_dim, hidden_size=self.dec_hidden_state_size,
                          num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.linear_layer = nn.Linear((2 * self.enc_hidden_state_size) + self.embedding_dim + self.dec_hidden_state_size, self.vocab_size)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, inp, dec_hidden_state, enc_outputs):
        # inp = [batch_size]
        # dec_hidden_state = [batch_size, dec_hidden_state]
        # enc_outputs = [src_seq_len, batch_size, 2*enc_hidden_state]
        
        inp = inp.unsqueeze(0)
        embedded = self.dropout_layer(self.embedding(inp))
        # embedded = [1, batch_size, embedding_dim]
        
        attn_weights = self.attn_layer(dec_hidden_state, enc_outputs).unsqueeze(1)
        # attn_weights = [batch_size, 1, src_seq_len]
        
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs = [batch_size, src_seq_len, 2*embedding_dim]
        
        weighted = torch.bmm(attn_weights, enc_outputs).squeeze(1).unsqueeze(0)
        # weighted = [1, batch_size, 2*embedding_dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch_size, 2*enc_hidden_state_size + embedding_dim]
        
        dec_outputs, dec_hidden_state = self.rnn(rnn_input, dec_hidden_state.unsqueeze(0))
        # dec_outputs == dec_hidden_state
        
        # dec_outputs = [1, batch_size, dec_hidden_state_size]
        # dec_hidden_state = [1, batch_size, dec_hidden_state_size]
        
        linear_layer_input = torch.cat((embedded.squeeze(0), weighted.squeeze(0), dec_outputs.squeeze(0)), dim=1)
        # linear_layer_input = [batch_size, 2*enc_hidden_state_size + embedding_dim + dec_hidden_state_size]
        
        outputs = self.linear_layer(linear_layer_input)
        # outputs = [batch_size, vocab_size]
        
        dec_hidden_state = dec_hidden_state.squeeze(0)
        
        return outputs, dec_hidden_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_sos_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_seq_len, batch_size]
        # trg = [trg_seq_len, batch_size]
        
        batch_size = src.shape[1]
        trg_seq_len = trg.shape[0]
        
        # final outputs from decoder
        final_outputs = torch.zeros((trg_seq_len, batch_size, self.decoder.vocab_size)).to(self.device)
        # setting first output as sos
        final_outputs[0, :, self.trg_sos_idx] = 0.98
        
        # encoder outputs
        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden_state = enc_hidden
        # print(enc_hidden.shape)
        dec_input = trg[0, :]
        for t in range(1, trg_seq_len):
            dec_outputs, dec_hidden_state = self.decoder(dec_input, dec_hidden_state, enc_outputs)
            final_outputs[t, :, :] = dec_outputs
            
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                dec_input = trg[t, :]
            else:
                dec_input = dec_outputs.max(1)[1]
        
        return final_outputs