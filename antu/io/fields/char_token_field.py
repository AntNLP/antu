from . import Field
import logging

logger = logging.getLogger(__name__)


class CharTokenField(Field):
    """Char token field: split each token into chars
    """

    def __init__(self, namespace, source_key):
        """This function set namespace name and dataset source key

        Arguments:
            namespace {str} -- namespace name
            source_key {str} -- indicate key in text data
        """

        self.namespace = namespace
        self.source_key = source_key
        super().__init__()

    def count_vocab_items(self, counter, sentences):
        """This function counts token's char in sentences,
        then updates counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            for token in sentence[self.source_key]:
                for char in token:
                    counter[self.namespace][str(char)] += 1

        logger.info(
            "Count sentences {} to update counter namespace {} successfully.".
            format(self.source_key, self.namespace))

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary,
        then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            token_num_repr = []
            for token in sentence[self.source_key]:
                token_num_repr.append([
                    vocab.get_token_index(char, self.namespace)
                    for char in token
                ])
            instance[self.namespace].append(token_num_repr)

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully."
            .format(self.source_key, self.namespace))
