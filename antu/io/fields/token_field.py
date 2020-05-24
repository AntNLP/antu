from . import Field
import logging

logger = logging.getLogger(__name__)


class TokenField(Field):
    """Token field: regard sentence as token list
    """

    def __init__(self, namespace, source_key):
        """This function set namespace name and dataset source key

        Arguments:
            namespace {str} -- namespace
            source_key {str} -- indicate key in text data
        """

        self.namespace = str(namespace)
        self.source_key = str(source_key)
        super().__init__()

    def count_vocab_items(self, counter, sentences):
        """This function counts tokens in sentences,
        then update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            for token in sentence[self.source_key]:
                counter[self.namespace][str(token)] += 1

        logger.info(
            "Count sentences {} to update counter namespace {} successfully.".
            format(self.source_key, self.namespace))

    def index(self, instance, vocab, sentences):
        """This function indexed token using vocabulary,
        then update instance
        
        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            instance[self.namespace].append([
                vocab.get_token_index(token, self.namespace)
                for token in sentence[self.source_key]
            ])

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully."
            .format(self.source_key, self.namespace))
