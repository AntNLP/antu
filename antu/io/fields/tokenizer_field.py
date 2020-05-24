from . import Field
import logging

logger = logging.getLogger(__name__)


class TokenizerField(Field):
    """This class using tokenizer to tokenize
    """

    def __init__(self, namespace, source_key, tokenizer):
        """This function set namespace name and dataset source key

        Arguments:
            namespace {str} -- namespace
            source_key {str} -- indicate key in text data
            tokenizer {Callable} -- tokenizer function
        """

        self.namespace = str(namespace)
        self.source_key = str(source_key)
        self.tokenizer = tokenizer
        super().__init__()

    def count_vocab_items(self, counter, sentences):
        """ `TokenizerField` doesn't update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        pass

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary,
        then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            tokenized_tokens = [vocab.get_token_index('[CLS]', self.namespace)]
            token_index = []
            index = 1
            for token in sentence[self.source_key]:
                tokenized_token = self.tokenizer(token)
                token_index.append(index)
                index += len(tokenized_token)
                tokenized_tokens.extend(
                    [vocab.get_token_index(item, self.namespace) for item in tokenized_token])
            tokenized_tokens.append(
                vocab.get_token_index('[SEP]', self.namespace))
            instance[self.namespace].append(tokenized_tokens)
            instance[self.namespace + '_index'].append(token_index)

        logger.info("Index sentences {} to construct instance namespace {} successfully.".format(
            self.source_key, self.namespace))
