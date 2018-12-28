import pytest
from antu.io.token_indexers.char_token_indexer import CharTokenIndexer
from antu.io.fields.text_field import TextField
from collections import Counter
from antu.io.vocabulary import Vocabulary

class TestCharTokenIndexer:

    def test_char_token_indexer(self):
        sentence = ['This', 'is', 'is', 'a', 'a', 'test', 'sentence']
        counter = {'my_char': Counter()}
        vocab = Vocabulary()
        glove = ['a', 'b', 'c', 'd', 'e']
        vocab.extend_from_pretrained_vocab({'glove': glove})
        indexer = CharTokenIndexer(['my_char', 'glove'])
        sent = TextField('sentence', sentence, [indexer])

        # Test count_vocab_items()
        sent.count_vocab_items(counter)
        assert counter['my_char']['s'] == 5
        assert counter['my_char']['T'] == 1
        assert counter['my_char']['t'] == 3
        assert counter['my_char']['A'] == 0

        vocab.extend_from_counter(counter)

        # Test index()
        sent.index(vocab)
        assert sent.indexes['glove'][0] == [0, 0, 0, 0] # 'This'
        assert sent.indexes['glove'][3] == [2]  # 'a'
        assert sent.indexes['my_char'][0] == [2, 3, 4, 5] # 'This'
