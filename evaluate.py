import data
import func
import utils
from collections import defaultdict


def evaluate_accuracy(model, dataset, batch_size=64, size=None, profile='dev'):
    model.eval()
    feeder = data.TrainFeeder(dataset)
    feeder.prepare(profile)
    size = size or feeder.size
    lines = []
    total_matches, total = defaultdict(lambda: 0), defaultdict(lambda: 0)
    output_file = f'./output/{profile}.txt'
    while feeder.cursor < size:
        stories, queries, _, tids, stories_text, queries_text, answers_text = feeder.next(batch_size)
        logits = model(func.tensor(stories), func.tensor(queries))
        _, predicts = logits.max(-1)
        for statements, query, answer, tid, predict in zip(stories_text, queries_text, answers_text, tids, predicts.tolist()):
            predict = feeder.id_to_word(predict)
            lines.append('------------------------------------')
            lines += statements
            lines.append(f'-> type:    {feeder.type(tid)}')
            lines.append(f'-> query:   {query}')
            lines.append(f'-> predict: {predict}')
            lines.append(f'-> answer:  {answer}')
            total_matches[tid] += 1 if answer == predict else 0
            total[tid] += 1
    for tid in total:
        accuracy = total_matches[tid] / total[tid]
        message = f'{tid}. {feeder.type(tid)}: {accuracy:>.2F}, Matches: {total_matches[tid]}, Total: {total[tid]}'
        lines.append(message)
        print(message)
    utils.write_all_lines(output_file, lines)
    return np.sum(total_matches.values()) / np.sum(total.values())

