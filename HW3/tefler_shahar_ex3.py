import random
from sys import argv
from os import path, listdir
from enum import Enum
from random import sample
import re

import lxml.etree
import numpy as np
import gender_guesser.detector as gender
from string import punctuation
from collections import Counter
from lxml import etree

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Gender(Enum):
    female = 1
    male = 2
    unknown = 3


CHUNK_SIZE = 10
gender_detector = gender.Detector()


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
    def __init__(self, sentence_id: str, authors: list, is_title=False, tokens: list = None):
        self.sentence_id = sentence_id
        self.is_title = is_title
        self.tokens = tokens
        self.author_names = authors

        # Defining gender of the sentence
        if len(authors) == 0:
            self.gender = Gender.unknown
        else:
            first_names = list()
            for name in authors:
                if "," not in name:
                    first_names.append(name)
                else:
                    first_names.append(name.split(",")[1].strip())

            authors_genders = self.filter_genders(list(map(gender_detector.get_gender, first_names)))
            if len(set(authors_genders)) == 1:
                self.gender = Gender[authors_genders[0]]
            else:
                self.gender = Gender.unknown

        if tokens is None:
            self.tokens = list()

    @staticmethod
    def filter_genders(genders: list) -> list:
        updated_genders = list()
        for gender in genders:
            if gender == "mostly_female" or gender == "mostly_male" or gender == "andy":
                updated_genders.append("unknown")
            else:
                updated_genders.append(gender)
        return updated_genders

    def add_token(self, token: Token):
        self.tokens.append(token)

    def count_punctuation_marks(self):
        """
        Counts how much punctuation marks are in the sentence.
        :return: Count of punctuation
        """
        counter = 0
        for token in self.tokens:
            if token.word in punctuation:
                counter += 1
        return counter

    def avg_word_length(self):
        return sum(map(lambda x: len(x.word), self.tokens)) / len(self.tokens)

    def __str__(self):
        try:
            return ' '.join(list(map(lambda x: x.word, self.tokens)))
        except Exception as e:
            print(e)


class MyFeatureVector:
    def __init__(self, chunk):
        # Counting average amount of tokens per sentence
        self.num_of_tokens = sum(map(lambda x: len(x.tokens), chunk.sentences)) / len(chunk.sentences)
        # Counting average amount of punctuation marks
        self.num_of_punctuations = sum(map(lambda x: x.count_punctuation_marks(), chunk.sentences)) / len(
            chunk.sentences)
        # Counting average word length
        self.avg_word_length = sum(map(lambda x: x.avg_word_length(), chunk.sentences)) / len(chunk.sentences)

        # Unique words
        dictionary = Counter()
        for sentence in chunk.sentences:
            dictionary.update(sentence.tokens)
        self.unique_word_ratio = len(dictionary.keys()) / sum(dictionary.values())

    def to_array(self) -> np.ndarray:
        return np.array([self.num_of_tokens, self.num_of_punctuations, self.avg_word_length, self.unique_word_ratio])


class Chunk:
    idx = 0

    def __init__(self, sentences: list = None):
        # Maintaining static index field for enumerating instances
        self.idx = Chunk.idx
        Chunk.idx += 1

        if sentences is None:
            self.sentences = list()
        else:
            self.sentences = sentences
        self.gender = self.sentences[0].gender

    def __str__(self):
        return ' '.join(list(map(lambda x: x.__str__(), self.sentences)))

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)


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

        self.chunks = list()

    def init_chunks(self):
        """
        Divides all the sentences in the corpus into chunks with a pre-defined size.
        :return: None
        """
        for sentence_idx in range(0, len(self.sentences), CHUNK_SIZE):
            curr_chunk = self.sentences[sentence_idx: sentence_idx + CHUNK_SIZE]
            self.chunks.append(Chunk(curr_chunk))

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

    @staticmethod
    def get_authors(xml_file: lxml.etree.Element):
        """
        Extracts authors of an XML document.
        :param xml_file: Input file
        :return: list of author names
        """
        return list(map(lambda x: x.text, list(xml_file.findall(".//author"))))


class Classify:
    def __init__(self, corpus: Corpus) -> None:
        """
        Initializes the classifier with the corpus and genders.
        """
        self.corpus = corpus
        self.corpus.init_chunks()
        self.chunks = list(filter(lambda x: x.gender != Gender.unknown, self.corpus.chunks))
        self.features = None
        self.my_features = None

    def count_genders(self) -> tuple:
        """
        Counts occurrences of each gender in the chunks.
        :return: female count, male count
        """
        female_count = sum(map(lambda x: int(x.gender == Gender.female), self.chunks))
        male_count = sum(map(lambda x: int(x.gender == Gender.male), self.chunks))
        return female_count, male_count

    def get_gender_chunks(self, gender: Gender) -> list:
        """
        Fetches all the chunks that are classified to the input gender.
        :param gender: input gender
        :return: list of all matching chunks
        """
        return list(filter(lambda x: x.gender == gender, self.chunks))

    def get_labels(self) -> list:
        """
        Fetches the label for each chunk in the corpus.
        :return: list of labels
        """
        return [chunk.gender.value for chunk in self.chunks]

    def bow_chunks(self) -> None:
        """
        Calculates the TF-IDF vector for all chunks
        :return: None
        """
        tfidf = TfidfVectorizer()
        # Concentrate all sentences in a chunk into a single string
        words = list(map(lambda x: x.__str__(), self.chunks))
        self.features = tfidf.fit_transform(words)
        self.my_features = [MyFeatureVector(chunk).to_array() for chunk in self.chunks]

    def down_sample(self) -> None:
        """
        Balances the count between the 2 genders defined.
        :return: None
        """
        female_count, male_count = self.count_genders()

        female_chunks = self.get_gender_chunks(Gender.female)
        male_chunks = self.get_gender_chunks(Gender.male)

        other_gender, gender_chunks, new_size = (
            male_chunks, female_chunks, male_count) if female_count > male_count else (
            female_chunks, male_chunks, female_count)

        self.chunks = other_gender + sample(gender_chunks, new_size)

    @staticmethod
    def evaluation(data, labels) -> tuple:
        """
        Evaluates a model given a data and its corresponding labels.
        :param data: input data
        :param labels: matching labels
        :return: score of model, report
        """
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        score = np.mean(cross_val_score(knn_classifier, data, labels, cv=10))

        # Normal testing with test-train split
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(x_train, y_train)
        y_predict = knn_classifier.predict(x_test)
        report = classification_report(y_test, y_predict, target_names=["female", "male"])

        return score * 100, report

    def print_evaluation_to_file(self, output_file) -> None:
        """
        Prints the evaluation of each model (BoW, custom features) into the file in the defined format.
        :param output_file: file to output to
        :return:
        """
        female_count_before, male_count_before = classifier.count_genders()
        classifier.down_sample()
        female_count_after, male_count_after = classifier.count_genders()

        self.bow_chunks()
        labels = self.get_labels()

        bow_score, bow_report = self.evaluation(self.features, labels)
        custom_features_score, custom_features_report = self.evaluation(self.my_features, labels)

        output_file.write("Before Down-sampling:\n")
        output_file.write("Female: {}   Male: {}\n".format(female_count_before, male_count_before))
        output_file.write("After Down-sampling:\n")
        output_file.write("Female: {}   Male: {}\n".format(female_count_after, male_count_after))

        output_file.write("== BoW Classification ==\n")
        output_file.write("Cross Validation Accuracy: {:.3f}%\n".format(bow_score))
        output_file.write(bow_report + "\n\n")

        output_file.write("== Custom Feature Vector Classification ==\n")
        output_file.write("Cross Validation Accuracy: {:.3f}\n".format(custom_features_score))
        output_file.write(custom_features_report + "\n\n")


if __name__ == '__main__':
    xml_dir = argv[1]
    output_file = argv[2]

    corpus = Corpus()

    length = len(listdir(xml_dir))
    for counter, file in enumerate(listdir(xml_dir)):
        corpus.add_xml_file_to_corpus(path.join(xml_dir, file))
        print("file ({}/{})".format(counter + 1, length))

    classifier = Classify(corpus)

    # Formatting results to output file
    output_file = open(output_file, "w")
    classifier.print_evaluation_to_file(output_file)
    output_file.close()
