import pytest
from antu.io.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from antu.io.token_indexers.char_token_indexer import CharTokenIndexer
from antu.io.fields.text_field import TextField
from collections import Counter
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance

class TestInstance:

    def test_instance(self):
        sentence = ['This', 'is', 'is', 'a', 'a', 'test', 'sentence']
        counter = {'my_word': Counter(), 'my_char': Counter()}
        vocab = Vocabulary()
        glove = ['This', 'is', 'glove', 'sentence', 'vocabulary']
        vocab.extend_from_pretrained_vocab({'glove': glove})
        single_id = SingleIdTokenIndexer(['my_word', 'glove'])
        char = CharTokenIndexer(['my_char'])
        sent = TextField('sentence', sentence, [single_id, char])
        data = Instance([sent])

        # Test count_vocab_items()
        data.count_vocab_items(counter)
        assert counter['my_word']['This'] == 1
        assert counter['my_word']['is'] == 2
        assert counter['my_word']['That'] == 0
        assert counter['my_char']['s'] == 5
        assert counter['my_char']['T'] == 1
        assert counter['my_char']['t'] == 3
        assert counter['my_char']['A'] == 0

        vocab.extend_from_counter(counter)

        # Test index()
        result = data.index_fields(vocab)
        assert result['sentence']['glove'] == [2, 3, 3, 0, 0, 0, 5]
        assert result['sentence']['my_word'] == [2, 3, 3, 4, 4, 5, 6]
        assert result['sentence']['my_char'][0] == [2, 3, 4, 5] # 'This'
        assert result['sentence']['my_char'][1] == result['sentence']['my_char'][2]
        assert result['sentence']['my_char'][3] == result['sentence']['my_char'][4]