import pytest
from antu.io.fields.text_field import TextField


class TestTextField:

    def test_textfield(self):
        sentence = ['This', 'is', 'a', 'test', 'sentence', '.']
        sent = TextField('sentence', sentence)
        assert sent[0] == 'This'
        assert sent[-1] == '.'
