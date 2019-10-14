from . import Field
import logging

logger = logging.getLogger(__name__)


class MapTokenField(Field):
    """Map token field: preocess maping tokens
    """

    def __init__(self, namespace, source_key):
        """This function set namespace name and dataset source key

        Arguments:
            namespace {str} -- namespace
            source_key {str} -- indicate key in text data
        """

        self.namespace = namespace
        self.source_key = source_key
        super().__init__()

    def count_vocab_items(self, counter, sentences):
        """This function counts dict's values in sentences,
        then update counter, each sentence is a dict

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing, list of dict
        """

        for sentence in sentences:
            for value in sentence[self.source_key].values():
                counter[self.namespace][str(value)] += 1

        logger.info(
            "Count sentences {} to update counter namespace {} successfully.".
            format(self.source_key, self.namespace))

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary, then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            instance[self.namespace].append({
                key: vocab.get_token_index(value, self.namespace)
                for key, value in sentence[self.source_key].items()
            })

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully."
            .format(self.source_key, self.namespace))
