class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
        # print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
        # print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
        # print(output.shape, hidden.shape)
        return output, (hidden, cell)


class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
        # print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
        # print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
        # print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
        # print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
        # print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs# class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs


class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs


class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs# class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs


class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
#         print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
#         print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
#             print()
#             print(t)
#             print(decoder_inp)
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs


class EncoderRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, inp):
#         print(inp.shape)
        # [seq_len, batch_size]
        inp = self.dropout(self.embeddings(inp))
#         print(inp.shape)
        # [seq_len, batch_size, embed_size]
        output, (hidden, cell) = self.gru(inp)
#         print(output.shape, hidden.shape)
        return output, (hidden, cell)

h_n.view(num_layers, num_directions, batch, hidden_size)

class DecodeRnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers,
                dropout=0.2):
        super(DecodeRnn, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.LSTM(embedding_dim, hidden_size,bidirectional=False, num_layers=self.num_layers)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, inp, hidden, cell):
#         print("inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        inp = inp.unsqueeze(0)
#         print("after unsqueezing inp shape {} and hidden shape {} is.".format(inp.shape, hidden.shape))
        
        # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
#         print("embbedded {} is.".format(embedded.shape))
        
        # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.gru(embedded, (hidden, cell))
#         print("decoder output shape {} and hidden shape {} is.".format(output.shape, hidden.shape))
        
        prediction = self.linear(output.squeeze(0))
#         print("prediction shape {} is.".format(prediction.shape))
        
        return prediction, (hidden, cell)

# dec = DecodeRnn(output_lang.n_words, 32, 50, num_layers=2)

# y.shape

# dec_inp = y[0, :]
# dec_inp.shape

# dec_out, (dec_hidden, dec_cell) = dec(dec_inp, enc_hidden, enc_cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # hidden dim of encoder and decoder must be same 
        assert encoder.hidden_size == decoder.hidden_size
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        # print("batch_size is {}, max_len is {}, and trg_vocab_size is {}".format(batch_size, max_len, trg_vocab_size))
        
        # saving outputs from decode
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # print("Outputs initialised shape {}".format(outputs.shape))
        
        # getting encoder outputs
        enc_out, (decoder_hidden, decoder_cell) = self.encoder(src)
        # print("Encoder output {} and hidden {}".format(enc_out.shape, decoder_hidden.shape))
        
        # first input to decoder is always the <SOS> token
        decoder_inp = trg[0, :]
        outputs[0,:, 0] = 0.99
        
        for t in range(1, max_len):
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder(decoder_inp, decoder_hidden, decoder_cell)
            outputs[t] = decoder_out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_out.max(1)[1]
            decoder_inp = (trg[t] if teacher_force else top1)
        return outputs