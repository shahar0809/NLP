from sys import argv
from os import path, listdir
from collections import Counter
import re
from bs4 import BeautifulSoup


class Token:
    def __init__(self, word: str = "", is_punctuation: bool = False, is_title: bool = False, is_multiword: bool = False,
                 hw: str = "", c5: str = "",
                 pos: str = ""):
        self.c5 = c5
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


class Corpus:
    sentence_split_delimiters = ["?", "!", ";", ":", "-", "'", '"', ")", "(", "’", "‘"]
    abbreviations = ["U.S.", "M.A.", "v.", ".com"]

    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "
    title_delimiter = "="

    def __init__(self, sentences: list = None):
        self.tokens_counter = Counter()
        self.sentences = sentences
        if sentences is None:
            self.sentences = list()
        self.files = list()

    def add_token(self, sentence: Sentence, token: Token):
        sentence.add_token(token)
        self.tokens_counter.update(token.word)

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        self.files.append(file_name)
        xml_file = open(file_name, "r", encoding="utf-8")
        xml_file = BeautifulSoup(xml_file, "xml")

        sentences = xml_file.findAll("s", recursive=True)
        for sentence in sentences:
            curr_sentence = Sentence(sentence["n"])
            for word in sentence.findAll(["w", "c"], recursive=True):
                if word.name == "c":
                    self.add_token(curr_sentence,
                                   Token(word=word.text, is_punctuation=True, is_multiword=word.parent.name == "mw",
                                         c5=word["c5"]))
                else:
                    self.add_token(curr_sentence,
                                   Token(word=word.text, is_punctuation=False, is_multiword=word.parent.name == "mw",
                                         hw=word["hw"], c5=word["c5"],
                                         pos=word["pos"]))
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
                # Looping over all sentences in paragraph
                for sentence_id, curr_sentence in enumerate(self.split_to_sentences(curr_paragraph)):
                    if not self.is_empty(curr_sentence):
                        sentence = Sentence(sentence_id=str(sentence_id),
                                            is_title=self.title_delimiter in curr_sentence)

                        # Looping over all tokens in sentence
                        for curr_token in curr_sentence.split(self.token_delimiter):
                            curr_token = curr_token.replace(self.title_delimiter, "")
                            if not self.is_empty(curr_token):
                                self.add_token(sentence, Token(curr_token, self.is_title(curr_sentence), False))
                        self.sentences.append(sentence)

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

        regex_pattern = "(" + ''.join(
            map(re.escape, self.sentence_split_delimiters)) + ")"  # regex pattern to match all delimiters from list
        last_end = 0  # last index of the previous dot
        sentences = list()  # list of sentences

        # Loop over all appearances of dot (with whitespace)
        for appearance in re.finditer(r"\. ", content):
            curr_str = content[last_end: appearance.start() + 1]

            # Checks if the dot appearance is not any abbreviation
            is_not_abbreviation = [not curr_str.endswith(abbreviation) for abbreviation in self.abbreviations]

            # Add current interval only if it does not end with abbreviation
            if all(is_not_abbreviation):
                sentences.extend(re.split(regex_pattern, curr_str))
                last_end = appearance.end()

        return sentences

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

    def get_count(self, phrases: list):
        counter = 0
        for sentence in self.sentences:
            for token_index in range(len(sentence.tokens)):
                if sentence.tokens[token_index: token_index + len(phrases)] == phrases:
                    counter += 1
        return counter


# Implement an n-gram language model class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class):


class NGramModel:
    def __init__(self, n: int, smoothing_type, corpus: Corpus):
        self.n = n
        self.smoothing_type = smoothing_type
        self.corpus = corpus
        self.n_grams = list()
        self.ngram_counter = Counter()

        self.build_ngram()
        print("done")

    def add_ngram(self, ngram: list):
        self.n_grams.append(ngram)

        ngram_str = ''.join(token.word for token in ngram)
        self.ngram_counter[ngram_str] += 1

    def build_ngram(self):
        for sentence in self.corpus.sentences:
            if len(sentence.tokens) >= self.n:
                for token_index in range(len(sentence.tokens) - self.n):
                    curr_ngram = list()
                    for ngram_index in range(self.n):
                        curr_ngram.append(sentence.tokens[token_index + ngram_index])
                    self.add_ngram(curr_ngram)

    def get_phrase_probability(self, phrase: Sentence):
        """
        Calculates the probability that the given phrase will appear in a sentence using MLE.
        :param phrase: A given phrase
        :return: Probability of phrase
        """
        product = 1
        for token_index in range(len(phrase.tokens)):
            product *= self.get_token_probability(phrase.tokens[token_index],
                                                  phrase.tokens[token_index - self.n + 1: token_index - 1])
        return product

    def get_token_probability(self, token, phrases):
        return self.corpus.get_count(phrases + [token]) / self.corpus.get_count(phrases)

    def laplace_smoothing(self):
        pass


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    corpus = Corpus()
    # for xml_file in listdir(xml_dir):
    #     corpus.add_xml_file_to_corpus(path.join(xml_dir, xml_file))

    corpus.add_xml_file_to_corpus("XML_files/A1D.xml")

    bigram = NGramModel(2, "d", corpus)

    # Implement here your program:
    # 1. Create a corpus from the file in the given directory.
    # 2. Create a language model based on the corpus.
    # 3. Calculate and print onto the output file the first task, in the wanted format.
    # 4. Print onto the output file the results from the second task in the wanted format.
