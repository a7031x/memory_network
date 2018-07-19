#implement http://arxiv.org/abs/1503.08895 | https://arxiv.org/pdf/1503.08895.pdf
#reference https://github.com/domluna/memn2n | https://github.com/domluna/memn2n.git

import torch
import torch.nn as nn
import data
import numpy as np
import func

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_sentence_size, hops):
        self.hops = hops
        self.A = nn.Embedding(vocab_size, embedding_size, padding_idx=data.PAD_ID)
        self.C = nn.ModuleList()
        self.encoding = self.position_encoding(max_sentence_size, embedding_size)
        for _ in range(hops):
            self.C.append(nn.Embedding(vocab_size, embedding_size, padding_idx=data.PAD_ID))


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
        q_emb = self.A(queries)
        u_0 = (q_emb * self.encoding).sum(1)
        u = [u_0]

        for hop in range(self.hops):
            if hop == 0:
                emb_A = self.A(stories)
            else:
                emb_A = self.C[hop-1](stories)
            A = (emb_A * self.encoding).sum(2)
            u_temp = u[-1].unsqueeze(-1).transpose(1, 2)
            dotted = (A * u_temp).sum(2)

            probs = nn.functional.softmax(dotted, -1)
            probs_temp = probs.unsqueeze(-1).transpose(1, 2)

            emb_C = self.C[hop](stories)
            C = (emb_C * self.encoding).sum(2)

            c_temp = C.transpose(1, 2)
            o_k = (c_temp * probs_temp).sum(2)

            u_k = u[-1] + o_k

            u.append(u_k)
        
        return torch.matmul(u_k, self.C[-1].transpose(0, 1))


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
    model = Model(dataset.vocab_size, opt.embedding_size, dataset.max_sentence_size, opt.hops)
    if func.gpu_available():
        model = model.cuda()
    return model


def build_train_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = build_model(opt, dataset)
    feeder = data.TrainFeeder(dataset)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
    feeder.prepare('train')
    return model, optimizer, feeder
