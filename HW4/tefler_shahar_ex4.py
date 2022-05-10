import re
from xml import etree

import lxml as lxml
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
from os import listdir, path
from typing import List, Tuple


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

    def __str__(self):
        return self.word.replace(" ", "")


class Sentence:
    def __init__(self, sentence_id: str, is_title=False, tokens: list = None):
        self.sentence_id = sentence_id
        self.is_title = is_title
        self.tokens = tokens
        if tokens is None:
            self.tokens = list()

    def add_token(self, token: Token):
        self.tokens.append(token)

    def __str__(self):
        try:
            return ' '.join(list(map(lambda x: x.word, self.tokens)))
        except Exception as e:
            print(e)


class Corpus:
    sentence_split_delimiters = ["?", "!", ";", ":", "-", "'", '"', ")", "(", "’", "‘"]
    abbreviations = ["U.S.", "M.A.", "v.", ".com"]

    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "
    title_delimiter = "="

    def __init__(self, sentences: list = None):
        self.sentences = sentences
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

        # sentences = xml_file.findAll("s", recursive=True)
        for sentence in xml_file.iter("s"):
            curr_sentence = Sentence(sentence.get("n"))
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
    pairs = [(("hot", "cold"), ("good", "bad")), (("day", "night"), ("right", "left")),
             (("cool", "cold"), ("warm", "hot")), (("tall", "short"), ("heavy", "thin")),
             (("like", "love"), ("dislike", "hate"))]

    words = list()
    for (pair, arithmetic_pair) in pairs:
        words += [model.most_similar(positive=[pair[0], arithmetic_pair[0]], negative=[arithmetic_pair[1]])]

    return pairs, words


def cos_distance(model: KeyedVectors, pair: Tuple[str, str]) -> float:
    return model.similarity(*pair)


def format_output(model, output):
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
        output.write("{}. {}:{}, {}:{}\n".format(idx + 1, *analogy))

    output.write("\nMost Similar:\n")
    for idx, (analogy, complement) in enumerate(pairs):
        output.write("{}. {} + {} - {} = {}\n".format(idx + 1, analogy[0], complement[0], complement[1]))


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
    # corpus = Corpus()
    # for file in listdir(xml_dir):
    #     corpus.add_xml_file_to_corpus(path.join(xml_dir, file))
