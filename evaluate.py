import data
import func
import utils
from collections import Counter


def evaluate_accuracy(model, dataset, batch_size=64, size=None, profile='dev'):
    model.eval()
    feeder = data.TrainFeeder(dataset)
    feeder.prepare(profile)
    size = size or feeder.size
    lines = []
    total_matches, total = 0, 0
    output_file = f'./output/{profile}.txt'
    while feeder.cursor < size:
        stories, queries, answers, stories_text, queries_text, answers_text = feeder.next(batch_size)
        logits = model(func.tensor(stories), func.tensor(queries))
        _, predicts = logits.max(-1)
        matches = (predicts == func.tensor(answers)).sum().tolist()
        total_matches += matches
        total += len(answers)
        for statements, query, answer in zip(stories_text, queries_text, answers_text):
            lines.append('------------------------------------')
            lines += statements
            lines.append(f'-> query:  {query}')
            lines.append(f'-> answer: {answer}')
        #print(f'{feeder.cursor}/{size}')
    accuracy = total_matches / total
    message = f'ACCURACY: {accuracy:>.2F}, Matches: {total_matches}, Total: {total}'
    lines.append(message)
    utils.write_all_lines(output_file, lines)
    print('evauation finished with ' + message)
    return accuracy

