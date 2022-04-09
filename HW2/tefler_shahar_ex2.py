from sys import argv
from os import path, listdir
from math import log2
from collections import Counter, deque
import re
from bs4 import BeautifulSoup
from itertools import count


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
    def __init__(self, sentence_id: str = 0, is_title=False, tokens: list = None):
        self.sentence_id = sentence_id
        self.is_title = is_title
        self.tokens = tokens
        if tokens is None:
            self.tokens = list()

    def add_token(self, token: Token):
        self.tokens.append(token)

    @staticmethod
    def make_sentence(words: str):
        return Sentence(tokens=[Token(word=word) for word in list(filter((" ").__ne__, re.split(r"([,.\s])", words)))])

    def __str__(self):
        return ' '.join([token.word for token in self.tokens])


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

    def __len__(self):
        return sum(len(sentence.tokens) for sentence in self.sentences)

    def add_token(self, sentence: Sentence, token: Token):
        sentence.add_token(token)
        self.tokens_counter[token.word.lower()] += 1

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


class NGram:
    def __init__(self, tokens: list):
        self.prev_tokens = tokens[:len(tokens) - 1]
        self.token = tokens[-1]

        self.ngram = ''.join(token.word for token in tokens)

    def __str__(self):
        return self.ngram


class LanguageModel:
    def __init__(self, n: int, corpus: Corpus, smoothing_type: str = "Laplace"):
        if smoothing_type != "Laplace" and smoothing_type != "Linear interpolation":
            raise Exception("Invalid smoothing type")

        self.n = n
        self.smoothing_type = smoothing_type
        self.corpus = corpus
        self.vocabulary = Counter()

        self.ngrams = Counter()  # Counter for n-grams
        self.contexts = Counter()  # Counter for (n-1)-grams

        self.build_vocabulary()

    def build_vocabulary(self):
        for sentence in self.corpus.sentences:
            for token in sentence.tokens:
                self.add_to_vocabulary(NGram([token]))

    def add_to_vocabulary(self, ngram: NGram):
        self.vocabulary[ngram.__str__().strip()] += 1

    def add_ngram(self, ngram: NGram):
        self.ngrams[ngram.__str__().strip()] += 1

    def add_context(self, context: list):
        self.contexts[self.concat_ngram(context).strip()] += 1

    def build_ngrams(self):
        for sentence in self.corpus.sentences:
            for ngram in self.n_wise(sentence.tokens, self.n):
                self.add_ngram(NGram(list(ngram)))
            for context in self.n_wise(sentence.tokens, self.n - 1):
                self.add_context(list(context))

    def get_phrase_probability(self, phrase: Sentence):
        """
        Calculates the probability that the given phrase will appear in a sentence using MLE.
        :param phrase: A given phrase
        :return: Probability of phrase
        """

        product = 1
        for token_index in range(self.n - 1, len(phrase.tokens)):
            product *= self.get_token_probability(phrase.tokens[token_index],
                                                  phrase.tokens[token_index - self.n + 1: token_index])
        return product

    def get_token_probability(self, token, phrases):
        """
        Calculates the probability of a token given the previous tokens.
        :param token: given token
        :param phrases: previous n-gram
        :return: probability as mentioned
        """
        if self.smoothing_type == "Laplace":
            return (self.get_count(phrases + [token]) + 1) / (self.get_count(phrases) + len(self.vocabulary))
        elif self.smoothing_type == "Linear interpolation":
            return self.linear_interpolation(token, phrases)

    def no_smoothing_probability(self, token, phrases):
        try:
            return self.get_count(phrases + [token]) / self.get_count(phrases)
        except ZeroDivisionError:
            return 0.5

    def linear_interpolation(self, token, phrases):
        raise NotImplementedError

    def get_count(self, phrases: list):
        """
        Calculates the count of appearances of a sequence of tokens in the NGram.
        :param phrases: sequence of tokens
        :return: count of appearances
        """
        if len(phrases) == self.n:
            return self.ngrams[self.concat_ngram(phrases).strip()]
        elif len(phrases) == self.n - 1:
            return self.contexts[self.concat_ngram(phrases).strip()]
        else:
            raise Exception("Invalid length at ngram")

    @staticmethod
    def concat_ngram(ngram: list):
        return ''.join(token.word for token in ngram)

    @staticmethod
    def n_wise(it, n: int):
        deq = deque((), n)
        for x in it:
            deq.append(x)
            if len(deq) == n: yield deq

    def calc_weights(self):
        weights = list()
        divisor = float(sum(range(1, self.n + 1)))
        for idx in range(1, self.n + 1):
            weights.append(idx / divisor)
        return weights


class UnigramModel(LanguageModel):
    def __init__(self, corpus: Corpus, smoothing_type: str):
        super().__init__(1, corpus, smoothing_type)
        self.ngrams = self.vocabulary

    def get_token_probability(self, token, phrases):
        """
        Calculates the probability of a token given the previous tokens.
        :param token: given token
        :param phrases: previous n-gram
        :return: probability as mentioned
        """
        if self.smoothing_type == "Laplace":
            return (self.get_count([token]) + 1) / (len(self.ngrams) + len(self.corpus))
        elif self.smoothing_type == "Linear interpolation":
            return self.get_count([token]) / len(self.ngrams)

    def linear_interpolation(self, token, phrases):
        return self.get_token_probability(token, phrases)


class BigramModel(LanguageModel):
    def __init__(self, corpus: Corpus, smoothing_type: str):
        super().__init__(2, corpus, smoothing_type)
        self.unigram = UnigramModel(corpus, smoothing_type)
        self.build_ngrams()

    def linear_interpolation(self, token, phrases):
        weights = self.calc_weights()
        return weights[0] * self.unigram.get_token_probability(token, list()) + \
               weights[1] * self.no_smoothing_probability(token, phrases[-1:])


class TrigramModel(LanguageModel):
    def __init__(self, corpus: Corpus, smoothing_type: str):
        super().__init__(3, corpus, smoothing_type)
        self.unigram = UnigramModel(corpus, smoothing_type)
        self.bigram = BigramModel(corpus, smoothing_type)

        self.build_ngrams()

    def linear_interpolation(self, token, phrases):
        weights = self.calc_weights()
        return weights[0] * self.unigram.get_token_probability(token, list()) + \
               weights[1] * self.bigram.no_smoothing_probability(token, phrases[-1:]) + \
               weights[2] * self.no_smoothing_probability(token, phrases[-2:])


class NGramModel(LanguageModel):
    def __init__(self, n: int, corpus: Corpus, smoothing_type: str):
        if n <= 3:
            raise Exception("NGramModel is for n > 3")

        super().__init__(n, corpus, smoothing_type)

        self.unigram = UnigramModel(corpus, smoothing_type)
        self.bigram = BigramModel(corpus, smoothing_type)
        self.trigram = TrigramModel(corpus, smoothing_type)

    def linear_interpolation(self, token, phrases):
        weights = self.calc_weights()
        return weights[0] * self.unigram.get_token_probability(token, phrases) + \
               weights[1] * self.bigram.get_token_probability(token, phrases) + \
               weights[2] * self.trigram.get_token_probability(token, phrases)


def part1(corpus: Corpus):
    unigram = UnigramModel(corpus, smoothing_type="Laplace")
    bigram = BigramModel(corpus, smoothing_type="Laplace")
    trigram = TrigramModel(corpus, smoothing_type="Linear interpolation")

    def print_model_stats(model: LanguageModel, sentences: list):
        for sentence in sentences:
            print(sentence.__str__())
            print("Probability: {}".format(log2(model.get_phrase_probability(sentence))))
            print("\n")

    sentences = [Sentence.make_sentence("May the Force be with you.")]
    sentences += [Sentence.make_sentence("I’m going to make him an offer he can’t refuse.")]
    sentences += [Sentence.make_sentence("Ogres are like onions.")]
    sentences += [Sentence.make_sentence("You’re tearing me apart, Lisa!")]
    sentences += [Sentence.make_sentence("I live my life one quarter at a time.")]

    print("*** Sentence Predictions ***\n")
    print("Unigrams Model\n")
    print_model_stats(unigram, sentences)

    print("\nBigrams Model\n")
    print_model_stats(bigram, sentences)

    print("\nTrigrams Model\n")
    print_model_stats(trigram, sentences)


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    corpus = Corpus()
    for xml_file in listdir(xml_dir):
        corpus.add_xml_file_to_corpus(path.join(xml_dir, xml_file))

    # corpus.add_xml_file_to_corpus("XML_files/A1D.xml")

    part1(corpus)
