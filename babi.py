import os
import re

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

    
def enumerate_dataset(folder):
    r = {}
    types = {}
    for filename in os.listdir(folder):
        def subs(span):
            return filename[span[0]:span[1]]

        m = re.search('qa(.*)_(.*)_(.*).txt', filename)
        tid = int(subs(m.span(1)))
        type = subs(m.span(2))
        dataset = subs(m.span(3))
        if tid in r:
            r[tid][dataset] = filename
        else:
            r[tid] = {dataset: filename}
        types[tid] = type
    for tid in sorted(types.keys()):
        train_path = os.path.join(folder, r[tid]['train'])
        test_path = os.path.join(folder, r[tid]['test'])
        type = types[tid]
        yield train_path, test_path, tid, type