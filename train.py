import options
import argparse
import evaluate
import models
import random
import func
import utils
from torch.nn.utils import clip_grad_norm_


def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    options.data_opts(parser)
    return parser.parse_args()


def run_epoch(opt, model, feeder, optimizer):
    model.train()
    criterion = models.make_loss_compute()
    total_loss = 0
    while True:
        stories, queries, answers, _, _, _, _ = feeder.next(opt.batch_size)
        logits = model(func.tensor(stories), func.tensor(queries))
        loss = criterion(logits, func.tensor(answers))
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        optimizer.step()
        total_loss += loss.tolist()
        if feeder.eof():
            break
    print(f'------ITERATION {feeder.iteration}, loss: {total_loss:>.4F}')
    return total_loss


class Logger(object):
    def __init__(self, opt):
        self.output_file = opt.summary_file
        self.lines = list(utils.read_all_lines(self.output_file))
        self.max_lines = opt.max_log_size


    def __call__(self, message):
        print(message)
        self.lines.append(message)
        self.lines = self.lines[:self.max_lines]
        utils.write_all_lines(self.output_file, self.lines)


def train(steps=400, evaluate_size=None):
    func.use_last_gpu()
    opt = make_options()
    model, optimizer, feeder, _ = models.load_or_create_models(opt, True)
    log = Logger(opt)
    last_accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size)
    loss_lines, accuracy_lines = [], []
    for itr in range(60):
        loss = run_epoch(opt, model, feeder, optimizer)
        accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size)
        loss_lines.append(f'{loss:>.4F}')
        accuracy_lines.append(f'{accuracy:>.4F}')
        if accuracy > last_accuracy:
            #models.save_models(opt, model, optimizer, feeder)
            last_accuracy = accuracy
            log(f'ITERATION {feeder.iteration}. MODEL SAVED WITH ACCURACY {accuracy:>.2F}.')
        else:
            if random.randint(0, 4) == 0:
                models.restore(opt, model, optimizer, feeder)
                log(f'ITERATION {feeder.iteration}. MODEL RESTORED {accuracy:>.2F}/{last_accuracy:>.2F}.')
            else:
                log(f'ITERATION {feeder.iteration}. CONTINUE TRAINING {accuracy:>.2F}/{last_accuracy:>.2F}.')
        if feeder.iteration % 10 == 0:
            accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size, profile='test')
            log(f'=========ITERATION {feeder.iteration}. TEST ACCURACY: {accuracy:>.2F}===========')
        utils.write_all_lines('./output/loss.txt', loss_lines)
        utils.write_all_lines('./output/accuracy.txt', accuracy_lines)
train()