#implement http://arxiv.org/abs/1503.08895 | https://arxiv.org/pdf/1503.08895.pdf
#reference https://github.com/domluna/memn2n | https://github.com/domluna/memn2n.git

import torch
import torch.nn as nn
import data
import numpy as np
import func
import utils
import os
import rnn


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_sentence_size, hops, rnn_type='gru'):
        super(Model, self).__init__()
        self.hops = hops
        self.rnn_type = rnn_type
        self.A = nn.Embedding(vocab_size, embedding_size, padding_idx=data.PAD_ID)
        self.reset_parameters(self.A)
        self.encoding = self.position_encoding(max_sentence_size, embedding_size)
        num_layers = 2
        self.qrnn = rnn.RNNEncoder(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            type=rnn_type)
        self.srnn = rnn.RNNEncoder(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            type=rnn_type)
        self.crnn = rnn.RNNEncoder(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            type=rnn_type)
        gru_hidden_size = embedding_size*self.crnn.num_states//2
        self.gru = nn.GRUCell(gru_hidden_size, gru_hidden_size)
        self.dense_state = nn.Linear(gru_hidden_size, embedding_size, bias=False)
        self.reset_parameters(self.dense_state)
        assert self.encoding.requires_grad == False
        assert self.A.weight.requires_grad


    def reset_parameters(self, x):
        x.weight.data /= 2


    def run_state(self, rnn, emb, state):
        if emb.dim() == 4:
            n, m, l, d = emb.shape
            state = self.run_state(rnn, emb.view(n*m, l, d), state.view(rnn.num_states, n*m, -1) if state is not None else None)
            return state.view(rnn.num_states, n, m, -1)
        lengths = torch.ones(emb.shape[0], emb.shape[1])
        if self.rnn_type == 'lstm' and state is not None:
            state = state.chunk(2)
        _, state = rnn(emb, lengths, state)
        if self.rnn_type == 'lstm':
            state = torch.cat(state, 0)
        return state


    def position_encoding(self, sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        # Make position encoding of time words identity to avoid modifying them 
        encoding[:, -1] = 1.0
        return func.tensor(np.transpose(encoding))


    def restack(self, state):
        states = torch.unbind(state, 0)
        return torch.cat(states, -1)


    def stack(self, state):
        states = state.chunk(self.crnn.num_states, dim=-1)
        return torch.stack(states, 0)


    def forward(self, stories, queries):
        q_emb = self.A(queries)#[batch,slen,dim]
        #self.encoding = 1
        u_k = self.run_state(self.qrnn, q_emb * self.encoding, None)#[num_states, batch, dim]
        A = None
        for _ in range(self.hops):
            emb_A = self.A(stories)#[batch,mlen,slen,dim]
            A = self.run_state(self.srnn, emb_A * self.encoding, A)
            dotted = torch.einsum('bmd,bd->bm', [self.restack(A), self.restack(u_k)])#[batch,mlen]
            #mask = stories.sum(-1) != 0
            #dotted -= (1-mask.float()) * 10000
            probs = nn.functional.softmax(dotted, -1).clone()#[batch,mlen]
            #emb_C = self.C[hop](stories)#[batch,mlen,slen,dim]
            #C = (emb_C * self.encoding).sum(2)#[batch,mlen,dim]
            C = self.run_state(self.crnn, emb_A * self.encoding, None)#[num_states, batch, mlen, dim]
            o_k = torch.einsum('bmd,bm->bd', (self.restack(C), probs))
            #u_k = u_k + o_k
            u_k = self.stack(self.gru(o_k, self.restack(u_k)))
        u = self.dense_state(self.restack(u_k))
        return torch.matmul(u, self.A.weight.transpose(0, 1))


class SingleWordLoss(nn.Module):
    def __init__(self):
        super(SingleWordLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(size_average=False)


    def forward(self, logits, target):
        loss = self.criterion(logits, target)
        return loss


def make_loss_compute():
    criterion = SingleWordLoss()
    if func.gpu_available():
        criterion = criterion.cuda()
    return criterion


def build_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = Model(dataset.vocab_size, opt.embedding_size, dataset.sentence_size, opt.hops)
    if func.gpu_available():
        model = model.cuda()
    return model


def build_train_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = build_model(opt, dataset)
    feeder = data.TrainFeeder(dataset)
    optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam
    }
    optimizer = optimizers[opt.optimizer]([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
    feeder.prepare('train')
    return model, optimizer, feeder


def load_or_create_models(opt, train):
    if os.path.isfile(opt.ckpt_path):
        ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
        model_options = ckpt['model_options']
        for k, v in model_options.items():
            setattr(opt, k, v)
            print('-{}: {}'.format(k, v))
    else:
        ckpt = None
    if train:
        model, optimizer, feeder = build_train_model(opt)
    else:
        model = build_model(opt)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        if train:
            optimizer.load_state_dict(ckpt['optimizer'])
            feeder.load_state(ckpt['feeder'])
    if train:
        return model, optimizer, feeder, ckpt
    else:
        return model, ckpt


def restore(opt, model, optimizer, feeder):
    if not os.path.isfile(opt.ckpt_path):
        return
    ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if feeder is not None:
        feeder.load_state(ckpt['feeder'])


def save_models(opt, model, optimizer, feeder):
    #model_options = ['char_hidden_size', 'encoder_hidden_size', 'rnn_type']
    model_options = ['embedding_size', 'hops']
    model_options = {k:getattr(opt, k) for k in model_options}
    utils.ensure_folder(opt.ckpt_path)
    torch.save({
        'model':  model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'feeder': feeder.state(),
        'model_options': model_options
        }, opt.ckpt_path)
