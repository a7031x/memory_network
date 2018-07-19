
import utils
import random
import re
import numpy as np
from six.moves import range, reduce
from itertools import chain


PAD_ID = 0

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


class Dataset(object):
    def __init__(self, opt):
        self.train_set = parse_stories(utils.read_all_lines(opt.single_fact_train_file))
        self.test_set = parse_stories(utils.read_all_lines(opt.single_fact_test_file))
        data = self.train_set + self.test_set
        self.train_set, self.dev_set = self.train_set[:900], self.train_set[900:]

        vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        max_story_size = max(map(len, (s for s, _, _ in data)))
        #mean_story_size = int(np.mean([ len(s) for s, _, _ in data]))
        sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        query_size = max(map(len, (q for _, q, _ in data)))
        memory_size = min(opt.memory_size, max_story_size)

        # Add time words/indexes
        for i in range(memory_size):
            word_idx['time{}'.format(i+1)] = len(word_idx)

        vocab_size = len(word_idx) + 1 # +1 for nil word
        sentence_size = max(query_size, sentence_size) # for the position
        sentence_size += 1  # +1 for time words

        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.word_idx = word_idx
        self.memory_size = memory_size
        self.i2w = {k:v for v,k in self.word_idx.items()}

    
class Feeder(object):
    def __init__(self, dataset):
        self.dataset = dataset


    def vectorize_data(self, data):
        """
        Vectorize stories and queries.

        If a sentence length < sentence_size, the sentence will be padded with 0's.

        If a story length < memory_size, the story will be padded with empty memories.
        Empty memories are 1-D arrays of length sentence_size filled with 0's.

        The answer array is returned as a one-hot encoding.
        """
        word_idx = self.dataset.word_idx
        sentence_size = self.dataset.sentence_size
        memory_size = self.dataset.memory_size
        S = []
        Q = []
        A = []
        for story, query, answer in data:
            ss = []
            for i, sentence in enumerate(story, 1):
                ls = max(0, sentence_size - len(sentence))
                ss.append([word_idx[w] for w in sentence] + [0] * ls)

            # take only the most recent sentences that fit in memory
            ss = ss[::-1][:memory_size][::-1]

            # Make the last word of each sentence the time 'word' which 
            # corresponds to vector of lookup table
            for i in range(len(ss)):
                ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

            lq = max(0, sentence_size - len(query))
            q = [word_idx[w] for w in query] + [0] * lq

            y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
            for a in answer:
                y = word_idx[a]

            S.append(ss)
            Q.append(q)
            A.append(y)
        return np.array(S, dtype=np.int64), np.array(Q, dtype=np.int64), np.array(A, dtype=np.int64)


class TrainFeeder(Feeder):
    def __init__(self, dataset):
        super(TrainFeeder, self).__init__(dataset)


    def prepare(self, type):
        if type == 'train':
            self.prepare_data(self.dataset.train_set)
            self.shuffle_index()
        elif type == 'dev':
            self.prepare_data(self.dataset.dev_set)
        else:
            self.prepare_data(self.dataset.test_set)
        self.cursor = 0
        self.iteration = 1


    def prepare_data(self, dataset):
        self.data = dataset
        self.data_index = list(range(len(self.data)))
        self.size = len(self.data)


    def state(self):
        return self.iteration, self.cursor, self.data_index


    def load_state(self, state):
        self.iteration, self.cursor, self.data_index = state


    def shuffle_index(self):
        random.shuffle(self.data_index)


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size):
        if self.eof():
            self.iteration += 1
            self.cursor = 0
            if self.data == self.dataset.train_set:
                self.shuffle_index()

        size = min(self.size - self.cursor, batch_size)
        batch = self.data_index[self.cursor:self.cursor+size]
        batch = [self.data[idx] for idx in batch]
        stories, queries, answers = zip(*batch)
        stories = [[' '.join(x) for x in y] for y in stories]
        queries = [' '.join(x) for x in queries]
        answers = [' '.join(x) for x in answers]
        S, Q, A = self.vectorize_data(batch)
        self.cursor += size
        return S, Q, A, stories, queries, answers

                
    def id_to_word(self, word):
        return self.dataset.i2w[word]
    

def align1d(value, mlen, fill=0):
    return value + [fill] * (mlen - len(value))


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [align1d(row, mlen, fill) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values


def align(values, fill=0):
    dim = 0
    inp = values
    while isinstance(inp, list):
        dim += 1
        inp = inp[0]
    if dim == 1:
        return values
    elif dim == 2:
        return align2d(values, fill)
    elif dim == 3:
        return align3d(values, fill)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    import argparse
    import options
    parser = argparse.ArgumentParser()
    options.data_opts(parser)
    options.train_opts(parser)
    opt = parser.parse_args()
    dataset = Dataset(opt)
    feeder = TrainFeeder(dataset)
    feeder.prepare('train')
    feeder.next(4)