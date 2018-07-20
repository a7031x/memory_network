#implement http://arxiv.org/abs/1503.08895 | https://arxiv.org/pdf/1503.08895.pdf
#reference https://github.com/domluna/memn2n | https://github.com/domluna/memn2n.git

import torch
import torch.nn as nn
import data
import numpy as np
import func
import utils
import os


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_sentence_size, hops):
        super(Model, self).__init__()
        self.hops = hops
        self.A = nn.Embedding(vocab_size, embedding_size, padding_idx=data.PAD_ID)
        self.reset_parameters(self.A)
        self.C = nn.ModuleList()
        self.encoding = self.position_encoding(max_sentence_size, embedding_size)
        for _ in range(hops):
            C = nn.Embedding(vocab_size, embedding_size, padding_idx=data.PAD_ID)
            self.reset_parameters(C)
            self.C.append(C)
        assert self.encoding.requires_grad == False
        assert self.A.weight.requires_grad
        for embedding in self.C:
            assert embedding.weight.requires_grad


    def reset_parameters(self, x):
        x.weight.data.normal_(0, 0.1)
        x.weight.data[x.padding_idx].fill_(0)


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


    def forward(self, stories, queries):
        q_emb = self.A(queries)#[batch,slen,dim]
        #self.encoding = 1
        u_k = (q_emb * self.encoding).sum(1)#[batch, dim]

        for hop in range(self.hops):
            if hop == 0:
                emb_A = self.A(stories)#[batch,mlen,slen,dim]
            else:
                emb_A = self.C[hop-1](stories)
            A = (emb_A * self.encoding).sum(2)#[batch, mlen, dim]
            dotted = torch.einsum('bmd,bd->bm', (A, u_k))#[batch,mlen]
            #mask = stories.sum(-1) != 0
            #dotted -= (1-mask.float()) * 10000
            probs = nn.functional.softmax(dotted, -1).clone()#[batch,mlen]

            emb_C = self.C[hop](stories)#[batch,mlen,slen,dim]
            C = (emb_C * self.encoding).sum(2)#[batch,mlen,dim]

            o_k = torch.einsum('bmd,bm->bd', (C, probs))
            u_k = u_k + o_k
        
        return torch.matmul(u_k, self.C[-1].weight.transpose(0, 1))


    def load(self, filename):
        import pickle
        with open(filename, 'rb') as file:
            nps = pickle.load(file)
            self.A = nn.Embedding.from_pretrained(func.tensor(nps['A']), freeze=False)
            self.A.padding_idx = data.PAD_ID
            C = nps['C']
            for i in range(len(C)):
                self.C[i] = nn.Embedding.from_pretrained(func.tensor(C[i]), freeze=False)
                self.C[i].padding_idx = data.PAD_ID
                assert self.C[i].weight.requires_grad
            assert self.A.weight.requires_grad


class SingleWordLoss(nn.Module):
    def __init__(self):
        super(SingleWordLoss, self).__init__()
        self.lsm = nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss(size_average=True)


    def forward(self, logits, target):
        output = self.lsm(logits)
        loss = self.criterion(output, target)
        return loss


def make_loss_compute():
    criterion = SingleWordLoss()
    if func.gpu_available():
        criterion = criterion.cuda()
    return criterion


def build_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = Model(dataset.vocab_size, opt.embedding_size, dataset.sentence_size, opt.hops)
    if os.path.isfile('./export.pkl')
        model.load('./export.pkl')
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
