import re
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# Chat abbreviation dictionary (same as training)
chat_words = {
    'ive': 'i have',
    'couldnt': 'could not',
    'id': 'i had',
    'im': 'i am',
    'dont': 'do not',
    'its': 'it is',
    'ur': 'your',
    'asap': 'as soon as possible',
    'isnt': 'is not',
    'thx': 'thanks',
    'ig': 'i guess',
    'nyc': 'New York',
    'm': 'am',
    'omg': 'oh my god',
    'u': 'you'
}

lemmatizer = WordNetLemmatizer()
exclude = string.punctuation


def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub('', text)


def remove_html_tags(text):
    if isinstance(text, str):
        return re.sub(r'<.*?>', '', text)
    return text


def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii')


def chat_conversion(text):
    new_text = []

    for word in text.split():
        if word.lower() in chat_words:
            new_text.append(chat_words[word.lower()])
        else:
            new_text.append(word)

    return " ".join(new_text)


def remove_punc1(text):
    return text.translate(str.maketrans('', '', exclude))


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    return " ".join([
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in text.split()
    ])


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_url(text)
    text = remove_html_tags(text)
    text = remove_emoji(text)
    text = chat_conversion(text)
    text = remove_punc1(text)
    text = lemmatize_text(text)

    return text