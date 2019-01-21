import argparse, _pickle, math, os, random, sys, time, logging
random.seed(666)
import numpy as np
np.random.seed(666)
from collections import Counter
from antu.io.vocabulary import Vocabulary
from antu.io.ext_embedding_readers import glove_reader
from antu.io.datasets.single_task_dataset import DatasetSetting, SingleTaskDataset
from utils.conllu_reader import PTBReader


def main():
    # Configuration file processing
    ...

    # DyNet setting
    ...

    # Build the dataset of the training process
    ## Build data reader
    data_reader = PTBReader(
        field_list=['word', 'tag', 'head', 'rel'],
        root='0\t**root**\t_\t**rpos**\t_\t_\t0\t**rrel**\t_\t_',
        spacer=r'[\t]',)
    ## Build vocabulary with pretrained glove
    vocabulary = Vocabulary()
    g_word, _ = glove_reader(cfg.GLOVE)
    pretrained_vocabs = {'glove': g_word}
    vocabulary.extend_from_pretrained_vocab(pretrained_vocabs)
    ## Setup datasets
    datasets_settings = {
        'train': DatasetSetting(cfg.TRAIN, True),
        'dev': DatasetSetting(cfg.DEV, True),
        'test': DatasetSetting(cfg.TEST, True),}
    datasets = SingleTaskDataset(vocabulary, datasets_settings, data_reader)
    counters = {'word': Counter(), 'tag': Counter(), 'rel': Counter()}
    datasets.build_dataset(
        counters, no_pad_namespace={'rel'}, no_unk_namespace={'rel'})

    # Build model
    ...

    # Train model
    train_batch = datasets.get_batches('train', cfg.TRAIN_BATCH_SIZE, True, cmp, True)
    valid_batch = datasets.get_batches('dev', cfg.TEST_BATCH_SIZE, True, cmp, False)
    test_batch  = datasets.get_batches('test', cfg.TEST_BATCH_SIZE, True, cmp, False)
    


if __name__ == '__main__':
    main()