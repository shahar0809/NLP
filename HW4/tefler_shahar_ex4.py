import re
from os import listdir, path
from sys import argv
from typing import List, Tuple, Callable
from xml import etree
from random import uniform
from sklearn.decomposition import PCA

import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt

VECTOR_DIMENSION = 50


class Token:
    def __init__(self, word: str = "", is_punctuation: bool = False, is_title: bool = False, is_multiword: bool = False,
                 hw: str = "", c5: str = "",
                 pos: str = ""):
        self.c5 = c5
        if word is None:
            self.word = ""
        else:
            self.word = word
        self.hw = hw
        self.pos = pos
        self.is_title = is_title
        self.is_multiword = is_multiword
        self.is_punctuation = is_punctuation
        self.word_vector = None

    def __str__(self):
        return self.word.replace(" ", "")


class Sentence:
    def __init__(self, sentence_id: str, is_title=False, tokens: list = None):
        self.sentence_id = sentence_id
        self.is_title = is_title
        self.tokens = tokens
        if tokens is None:
            self.tokens = list()
        self.word_vector = None

    def add_token(self, token: Token):
        self.tokens.append(token)

    def __str__(self):
        try:
            return ' '.join(list(map(lambda x: x.word, self.tokens)))
        except Exception as e:
            print(e)


class Tweet:
    def __init__(self, idx: int, category: str):
        self.idx = idx
        self.sentences = list()
        self.category = category

    def set_sentences(self, sentences: List[Sentence]):
        self.sentences = sentences

    def tweet_vector(self, model: KeyedVectors, weight_func: Callable) -> np.ndarray:
        tweet_vector = np.zeros(VECTOR_DIMENSION)
        for sentence in self.sentences:
            sentence.word_vector = vectors_average(sentence.tokens, weight_func, model)
            tweet_vector += sentence.word_vector
        return tweet_vector / len(self.sentences)


class Corpus:
    sentence_split_delimiters = ["?", "!", ";", ":", "-", "'", '"', ")", "(", "’", "‘"]
    abbreviations = ["U.S.", "M.A.", "v.", ".com"]

    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "
    title_delimiter = "="

    def __init__(self, sentences: list = None):
        self.sentences = sentences
        self.tweets = list()
        if sentences is None:
            self.sentences = list()
        self.files = list()

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        self.files.append(file_name)
        xml_file = open(file_name, "r", encoding="utf-8")
        # xml_file = BeautifulSoup(xml_file, "lxml")

        xml_file = etree.parse(xml_file)
        authors = self.get_authors(xml_file)

        # sentences = xml_file.findAll("s", recursive=True)
        for sentence in xml_file.iter("s"):
            curr_sentence = Sentence(sentence.get("n"), authors)
            for word in list(sentence):
                curr_token = None
                if word.tag == "w":
                    curr_token = Token(word=word.text, is_punctuation=False,
                                       hw=word.get("hw"), c5=word.get("c5"),
                                       pos=word.get("pos"))
                elif word.tag == "c":
                    curr_token = Token(word=word.text, is_punctuation=True,
                                       c5=word.get("c5"))
                elif word.tag == "mw":
                    for sub_word in list(word):
                        if sub_word.tag == "w":
                            curr_token = Token(word=sub_word.text, is_punctuation=False,
                                               hw=sub_word.get("hw"), c5=sub_word.get("c5"),
                                               pos=sub_word.get("pos"))
                        elif sub_word.tag == "c":
                            curr_token = Token(word=sub_word.text, is_punctuation=True,
                                               c5=sub_word.get("c5"))
                if curr_token is not None:
                    curr_sentence.add_token(curr_token)
            if len(curr_sentence.tokens) != 0:
                self.sentences.append(curr_sentence)

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """

        self.files.append(file_name)
        text_file = open(file_name, "r", encoding="utf-8")
        text_file_content = text_file.read()

        # Looping over all paragraphs
        for curr_paragraph in text_file_content.split(self.paragraph_delimiter):
            if not self.is_empty(curr_paragraph):
                self.sentences = self.tokenize(curr_paragraph)

    def add_tweets_file_to_corpus(self, file_name: str):
        text_file = open(file_name, "r", encoding="utf-8")
        text_file_content = text_file.read()

        tweet_counter = 0
        for category in text_file_content.split("== ")[1:]:
            category_name = category[:category.find(" ==")]
            category = category.split(self.paragraph_delimiter)[1:]
            for tweet_text in category:
                if not self.is_empty(tweet_text):
                    tweet = Tweet(tweet_counter, category_name)
                    tweet.set_sentences(self.tokenize(tweet_text))
                    if len(tweet.sentences) != 0:
                        self.tweets.append(tweet)
                        tweet_counter += 1

    def tokenize(self, text: str) -> List[Sentence]:
        sentences = list()
        for sentence_id, curr_sentence in enumerate(filter(lambda x: len(x) != 0, self.split_to_sentences(text))):
            if not self.is_empty(curr_sentence):
                sentence = Sentence(sentence_id=str(sentence_id),
                                    is_title=self.title_delimiter in curr_sentence)

                # Looping over all tokens in sentence
                for curr_token in re.findall(r"\w+|[^\s\w]+", curr_sentence):
                    curr_token = curr_token.replace(self.title_delimiter, "")
                    if not self.is_empty(curr_token):
                        sentence.add_token(Token(curr_token, self.is_title(curr_sentence), False))
                sentences.append(sentence)
        return sentences

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """

        file = open(file_name, "wb")
        for sentence in self.sentences:
            tokens_strings = list()
            for token in sentence.tokens:
                tokens_strings.append(str(token))
            output = " ".join(tokens_strings) + self.paragraph_delimiter
            if sentence.is_title:
                output = ":title: " + output
            file.write(output.encode())
        file.close()

    def split_to_sentences(self, content):
        """
        Splits a segment of text into sentences, while considering abbreviations from a pre-defined list.
        :param content: The text segment
        :return: list of sentences
        """

        regex_pattern = r"(?<=[\)*?!.:;-])"

        # if "." not in content or content.find(".") == len(content) - 1:
        #     return re.split(regex_pattern, content)

        # Loop over all appearances of dot (with whitespace)
        # for appearance in re.finditer(r"\. ", content):
        #     curr_str = content[last_end: appearance.start() + 1]
        #
        #     # Checks if the dot appearance is not any abbreviation
        #     is_not_abbreviation = [not curr_str.endswith(abbreviation) for abbreviation in self.abbreviations]
        #
        #     # Add current interval only if it does not end with abbreviation
        #     if all(is_not_abbreviation):
        #         sentences.extend(re.split(regex_pattern, curr_str))
        #         last_end = appearance.end()

        return filter(lambda x: len(x) != 0, re.split(regex_pattern, content))

    @staticmethod
    def is_title(paragraph: str):
        """
        Checks if a string is a title in text documents.
        :param paragraph: the input
        :return: if it contains '=' character
        """
        return "=" in paragraph

    @staticmethod
    def is_empty(content: str):
        """
        Checks if a string is empty or contains only new lines.
        :param content: string
        :return: True if it's empty, false otherwise
        """
        return re.compile("[\\n\\r]+").match(content) or content == ""


def convert_to_word2vec(text_filename: str, kv_filename: str) -> None:
    """
    Converts a GloVe vector to a format that gensim can read (KeyVector)
    :param text_filename: A GloVe file
    :param kv_filename: Output KeyVector file
    :return: None
    """
    # Save the GloVe text file to a word2vec file for your use:
    glove2word2vec(text_filename, kv_filename)
    # Load the file as KeyVectors:
    model = KeyedVectors.load_word2vec_format(kv_filename, binary=False)
    # Save the key vectors for your use:
    model.save(kv_filename)


def load_key_vector(kv_filename: str) -> object:
    return KeyedVectors.load(kv_filename, mmap='r')


def word_pairs() -> List[Tuple[str, str]]:
    return [("hot", "cold"), ("good", "bad"), ("west", "east"), ("always", "never"), ("yes", "no"),
            ("apple", "banana"), ("kid", "child"), ("play", "act"), ("little", "tiny"), ("van", "car")]


def analogies(model: KeyedVectors) -> tuple:
    pairs = [(("hot", "cold"), ("love", "hate")), (("dog", "cat"), ("right", "left")),
             (("lighting", "thunder"), ("puppy", "dog")), (("north", "south"), ("boy", "girl")),
             (("like", "love"), ("dislike", "hate"))]

    words = list()
    for (pair, arithmetic_pair) in pairs:
        words += [model.most_similar(positive=[pair[0], arithmetic_pair[0]], negative=[arithmetic_pair[1]], topn=1)[0][
                      0]]

    return pairs, words


def cos_distance(model: KeyedVectors, pair: Tuple[str, str]) -> float:
    """
    Calculates the cosine distance between 2 words using the pre-trained model.
    :param model: word2vec model
    :param pair: A tuple containing the two input words
    :return: Cosine distance between the words
    """
    return model.similarity(*pair)


def get_vector(model: KeyedVectors, word: Token):
    try:
        return model[word.word.strip().lower()]
    except KeyError:
        return np.zeros(VECTOR_DIMENSION)


def word_similarities(model, output):
    # Word Pairs
    pairs = word_pairs()
    cos_dist = {pair: cos_distance(model, pair) for pair in pairs}

    output.write("Word Pairs and Distances:\n")
    for idx, pair in enumerate(cos_dist.keys()):
        output.write("{}. {} - {}: {:.3f}\n".format(idx + 1, pair[0], pair[1], cos_dist[pair]))

    # Analogies
    pairs, most_similar = analogies(model)
    output.write("\nAnalogies:\n")
    for idx, analogy in enumerate(pairs):
        output.write("{}. {}:{}, {}:{}\n".format(idx + 1, *analogy[0], *analogy[1]))

    output.write("\nMost Similar:\n")
    for idx, (analogy, complement) in enumerate(pairs):
        output.write(
            "{}. {} + {} - {} = {}\n".format(idx + 1, analogy[0], complement[0], complement[1], most_similar[idx]))

    output.write("\nDistances:\n")
    for idx, (analogy, complement) in enumerate(pairs):
        output.write(
            "{}. {} - {} : {}\n".format(idx + 1, analogy[1], most_similar[idx],
                                        cos_distance(model, (analogy[1], most_similar[idx]))))


def arithmetic_average(word: Token) -> float:
    return 1


def random_weights(word: Token) -> float:
    return uniform(0, 9.999999)


def vectors_average(words: List[Token], weight_func: Callable, model: KeyedVectors) -> np.ndarray:
    result = np.zeros(VECTOR_DIMENSION)
    for token in words:
        result += weight_func(token) * get_vector(model, token)
    return result / len(words)


def format_tweets(tweets, model: KeyedVectors):
    tweets_corpus = Corpus()
    tweets_corpus.add_tweets_file_to_corpus(tweets)

    for weight_func in [arithmetic_average, random_weights]:
        pca = PCA(n_components=2)
        embedded_vectors = np.array([tweet.tweet_vector(model, weight_func) for tweet in tweets_corpus.tweets])
        graph_points = pca.fit_transform(embedded_vectors)

        plt.title(weight_func.__name__)
        plt.scatter(graph_points[:, 0], graph_points[:, 1])
        plt.show()


def format_output(model, output):
    word_similarities(model, output)
    output.close()


if __name__ == "__main__":
    kv_file = argv[1]
    xml_dir = argv[2]
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]

    output_file = open(output_file, "w")

    # convert_to_word2vec(kv_file, path.join("word2vec", path.basename(kv_file)[:-4] + ".kv"))
    pre_trained_model = load_key_vector(kv_file)
    format_output(pre_trained_model, output_file)

    format_tweets(tweets_file, pre_trained_model)

    corpus = Corpus()
    # for file in listdir(xml_dir):
    #     corpus.add_xml_file_to_corpus(path.join(xml_dir, file))
