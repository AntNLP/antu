import pytest
from antu.io.vocabulary import Vocabulary
from collections import Counter


class TestVocabulary:

    def test_extend_from_pretrained_vocab(self):
        vocab = Vocabulary()

        # Test extend a vocabulary from a simple pretained vocab
        pretrained_vocabs = {'glove': ['a', 'b', 'c']}
        vocab.extend_from_pretrained_vocab(pretrained_vocabs)
        assert vocab.get_token_index('a', 'glove') == 2
        assert vocab.get_token_index('c', 'glove') == 4
        assert vocab.get_token_index('d', 'glove') == 0

        # Test extend a vocabulary from a pretained vocabulary,
        # and intersect with another vocabulary.
        pretrained_vocabs = {'w2v': ['b', 'c', 'd']}
        vocab.extend_from_pretrained_vocab(pretrained_vocabs, {'w2v':'glove'})
        assert vocab.get_token_index('b', 'w2v') == 2
        assert vocab.get_token_index('d', 'w2v') == 0
        assert vocab.get_token_from_index(2, 'w2v') == 'b'
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_from_index(4, 'w2v')
        assert excinfo.type == RuntimeError

        # Test extend a vocabulary from a no oov pretained vocabulary
        pretrained_vocabs = {'glove_nounk': ['a', 'b', 'c']}
        vocab.extend_from_pretrained_vocab(
            pretrained_vocabs, no_unk_namespace={'glove_nounk',})
        assert vocab.get_token_index('a', 'glove_nounk') == 1
        assert vocab.get_token_index('c', 'glove_nounk') == 3
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('d', 'glove_nounk')
        assert excinfo.type == RuntimeError

        # Test extend a vocabulary from a no oov and pad pretained vocabulary
        pretrained_vocabs = {'glove_nounk_nopad': ['a', 'b', 'c']}
        vocab.extend_from_pretrained_vocab(
            pretrained_vocabs,
            no_unk_namespace={'glove_nounk_nopad',},
            no_pad_namespace={"glove_nounk_nopad"})
        assert vocab.get_token_index('a', 'glove_nounk_nopad') == 0
        assert vocab.get_token_index('c', 'glove_nounk_nopad') == 2
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('d', 'glove_nounk_nopad')
        assert excinfo.type == RuntimeError

    def test_extend_from_counter(self):
        vocab = Vocabulary()

        # Test extend a vocabulary from a simple counter
        counter = {'w': Counter(["This", "is", "a", "test", "sentence", '.'])}
        vocab.extend_from_counter(counter)
        assert vocab.get_token_index('a', 'w') == 4
        assert vocab.get_token_index('.', 'w') == 7
        assert vocab.get_token_index('That', 'w') == 0

        # Test extend a vocabulary from a counter with min_count
        counter = {'w_m': Counter(['This', 'is', 'is'])}
        min_count = {'w_m': 2}
        vocab.extend_from_counter(counter, min_count)
        assert vocab.get_token_index('is', 'w_m') == 2
        assert vocab.get_token_index('This', 'w_m') == 0
        assert vocab.get_token_index('That', 'w_m') == 0

        # Test extend a vocabulary from a counter without oov token
        counter = {'w_nounk': Counter(['This', 'is'])}
        vocab.extend_from_counter(counter, no_unk_namespace={'w_nounk',})
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('That', 'w_nounk')
        assert excinfo.type == RuntimeError
        assert vocab.get_token_index('This', 'w_nounk') == 1

        # Test extend a vocabulary from a counter without pad & unk token
        counter = {'w_nounk_nopad': Counter(['This', 'is', 'a'])}
        vocab.extend_from_counter(
            counter,
            no_unk_namespace={'w_nounk_nopad'},
            no_pad_namespace={'w_nounk_nopad'})
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('That', 'w_nounk_nopad')
        assert excinfo.type == RuntimeError
        assert vocab.get_token_index('This', 'w_nounk_nopad') == 0

    def test_vocabulary(self):
        pretrained_vocabs = {
            'glove': ['a', 'b', 'c'],
            'w2v': ['b', 'c', 'd'],
            'glove_nounk': ['a', 'b', 'c'],
            'glove_nounk_nopad': ['a', 'b', 'c']}

        counters = {
            'w': Counter(["This", "is", "a", "test", "sentence", '.']),
            'w_m': Counter(['This', 'is', 'is']),
            'w_nounk': Counter(['This', 'is']),
            'w_nounk_nopad': Counter(['This', 'is', 'a'])}

        vocab = Vocabulary(
            counters=counters,
            min_count={'w_m': 2},
            pretrained_vocab=pretrained_vocabs,
            intersection_vocab={'w2v': 'glove'},
            no_pad_namespace={'glove_nounk_nopad', 'w_nounk_nopad'},
            no_unk_namespace={
                'glove_nounk', 'w_nounk', 'glove_nounk_nopad', 'w_nounk_nopad'})

        # Test glove
        print(vocab.get_vocab_size('glove'))
        assert vocab.get_token_index('a', 'glove') == 2
        assert vocab.get_token_index('c', 'glove') == 4
        assert vocab.get_token_index('d', 'glove') == 0

        # Test w2v
        assert vocab.get_token_index('b', 'w2v') == 2
        assert vocab.get_token_index('d', 'w2v') == 0
        assert vocab.get_token_from_index(2, 'w2v') == 'b'
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_from_index(4, 'w2v')
        assert excinfo.type == RuntimeError

        # Test glove_nounk
        assert vocab.get_token_index('a', 'glove_nounk') == 1
        assert vocab.get_token_index('c', 'glove_nounk') == 3
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('d', 'glove_nounk')
        assert excinfo.type == RuntimeError

        # Test glove_nounk_nopad
        assert vocab.get_token_index('a', 'glove_nounk_nopad') == 0
        assert vocab.get_token_index('c', 'glove_nounk_nopad') == 2
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('d', 'glove_nounk_nopad')
        assert excinfo.type == RuntimeError

        # Test w
        assert vocab.get_token_index('a', 'w') == 4
        assert vocab.get_token_index('.', 'w') == 7
        assert vocab.get_token_index('That', 'w') == 0

        # Test w_m
        assert vocab.get_token_index('is', 'w_m') == 2
        assert vocab.get_token_index('This', 'w_m') == 0
        assert vocab.get_token_index('That', 'w_m') == 0

        # Test w_nounk
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('That', 'w_nounk')
        assert excinfo.type == RuntimeError
        assert vocab.get_token_index('This', 'w_nounk') == 1

        # Test w_nounk_nopad
        with pytest.raises(RuntimeError) as excinfo:
            vocab.get_token_index('That', 'w_nounk_nopad')
        assert excinfo.type == RuntimeError
        assert vocab.get_token_index('This', 'w_nounk_nopad') == 0