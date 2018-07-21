import argparse

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-embedding_size', type=int, default=40)
    group.add_argument('-hops', type=int, default=3)
    group.add_argument('-ckpt_path', type=str, default='./checkpoint/model.pt')


def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-batch_size', type=int, default=32)
    group.add_argument('-learning_rate', type=float, default=1E-2)
    group.add_argument('-max_grad_norm', type=float, default=40.0)
    group.add_argument('-max_log_size', type=int, default=1000)
    group.add_argument('-optimizer', type=str, default='sgd')
    group.add_argument('-summary_file', type=str, default='./output/summary.txt')


def evaluate_opts(parser):
    group = parser.add_argument_group('evaluate')
    group.add_argument('-batch_size', type=int, default=64)


def data_opts(parser):
    group = parser.add_argument_group('data')
    group.add_argument('-memory_size', type=int, default=50)
    group.add_argument('-single_fact_train_file', type=str, default='./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')
    group.add_argument('-single_fact_test_file', type=str, default='./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    group.add_argument('-babi_en_folder', type=str, default='./data/tasks_1-20_v1-2/en')

