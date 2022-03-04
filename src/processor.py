# import pandas as pd
import json
import nltk
import re
import random
from collections import Counter
from typing import List, Dict, Tuple, Counter, Union

from . import config
# from src import config

# nltk.download('wordnet')
nltk.download('omw-1.4')

# import importlib
# importlib.reload(config)

# file_path = config.TRAINING_FILE

def load_data(file_path):

    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)

    print(f"There are {len(data_dict['data'])} twits in dataset.")
    print(f"A example of data: {data_dict['data'][0]}")

    ## Split message body and sentiment score
    messages = [data['message_body'] for data in data_dict['data']]

    # Since the sentiment scores are discrete, we'll scale the sentiments
    # form [-2 to 2] to [0 to 4] for use in our network
    sentiments = [data['sentiment'] + 2 for data in data_dict['data']]

    return messages, sentiments


def preprocess(message) -> List[str]:
    """
    This function takes a string as input, then performs these operations:
        - lowercase
        - remove URLs
        - remove ticker symbols
        - removes punctuation
        - tokenize by splitting the string on whitespace
        - removes any single character tokens

    Parameters:
        message: (str) The text message to be preprocessed.

    Returns:
        tokens: (list of words) The preprocessed text into tokens.
    """
    # Lowercase the twit message
    text = message.lower()

    # Replace URLs with a space in the message
    text = re.sub(r'http[s]?\S+', ' ', text)

    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub(r'\$\S+', ' ', text)

    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub(r'@\S+', ' ', text)

    # Replace everything not a letter with a space
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. We can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t, pos='v') for t in tokens]

    assert type(tokens) == list, 'Tokens should be list'

    return tokens


def rm_commom_words(bag_of_words: Counter[str], num_messages: int,
                    low_cutoff: float, high_cutoff: int) -> List[str]:
    # dict that contains the frequency of words appearing in messages
    """
    Args:
        low_cutoff: the frequency cutoff. Drop words with a freq <= low_cutoff
        high_cutoff: the cutoff for most common words. Drop words > high_cutoff
    """
    freqs = {token: (counts/num_messages) for token, counts in bag_of_words.items()}

    # The k most common words in the corpus. Use `high_cutoff` as the k.
    k_most_common = sorted(freqs, key=freqs.get, reverse=True)[:high_cutoff]

    # filter the low freqs words and k most common words
    filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in k_most_common)]

    return filtered_words


def vocab_mapping(words_list: List[str], cls_id: int, sep_id: int, pad_id: int
                  ) -> Tuple[Dict[str, int], Dict[int, str]]:

    id_to_vocab = {idx: word for idx, word in enumerate(words_list)}
    # Avoid occupy the ids of special tokens such as [CLS], [SEP], [PAD]
    id_to_vocab.update({len(words_list): id_to_vocab.get(cls_id),
                        len(words_list)+1: id_to_vocab.get(sep_id),
                        len(words_list)+2: id_to_vocab.get(pad_id)})
    id_to_vocab[cls_id] = '[CLS]'
    id_to_vocab[sep_id] = '[SEP]'
    id_to_vocab[pad_id] = '[PAD]'

    vocab_to_id = {word: idx for idx, word in id_to_vocab.items()}

    assert set(vocab_to_id.keys()) == set(id_to_vocab.values())

    return vocab_to_id, id_to_vocab


def balance_classes(messages: List[List[str]],
                    sentiments: List[int],
                    neutral_score: int
                    ) -> Dict[str, Union[List[str], List[int]]]:
    """
    Balance the neutral class. We should also take this opportunity to remove
    messages with length 0.
    """
    sentiments_count = Counter(sentiments)

    balanced = {'messages': [], 'sentiments': []}

    n_neutral = sentiments_count[neutral_score]
    n_examples = len(sentiments)

    # (n_examples - n_neutral)/4: the avg num of examples of each of remaining 4 class
    # except neutral, which is the number the neutral class should be rebalanced to.
    # (n_examples - n_neutral)/4/n_neutral: the ratio that the existing neutral class should keep
    keep_prob = (n_examples - n_neutral) / 4 / n_neutral

    for idx, sentiment in enumerate(sentiments):
        message = messages[idx]
        # skip the message if its length is zero
        if len(message) == 0:
            continue
        elif sentiment != 2 or random.random() < keep_prob:
            balanced['messages'].append(message)
            balanced['sentiments'].append(sentiment)

    return balanced


