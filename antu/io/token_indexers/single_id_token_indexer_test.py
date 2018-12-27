import pytest
from antu.io.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from antu.io.fields.text_field import TextField
from collections import Counter
from antu.io.vocabulary import Vocabulary

class TestSingleIdTokenIndexer:

    def test_single_id_token_indexer(self):
        sentence = ['This', 'is', 'is', 'a', 'a', 'test', 'sentence']
        counter = {'my_word': Counter()}
        vocab = Vocabulary()
        glove = ['This', 'is', 'glove', 'sentence', 'vocabulary']
        vocab.extend_from_pretrained_vocab({'glove': glove})
        indexer = SingleIdTokenIndexer(['my_word', 'glove'])
        sent = TextField('sentence', sentence, [indexer])

        # Test count_vocab_items()
        sent.count_vocab_items(counter)
        assert counter['my_word']['This'] == 1
        assert counter['my_word']['is'] == 2
        assert counter['my_word']['That'] == 0

        vocab.extend_from_counter(counter)

        # Test index()
        sent.index(vocab)
        assert sent.indexes['glove'] == [2, 3, 3, 0, 0, 0, 5]
        assert sent.indexes['my_word'] == [2, 3, 3, 4, 4, 5, 6]