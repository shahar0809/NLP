from sys import argv
from os import path
from os import listdir
from enum import Enum

import re
from bs4 import BeautifulSoup
import gender_guesser.detector as gender


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
        for sentence_idx in range(len(self.sentences) // CHUNK_SIZE):
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
        xml_file = BeautifulSoup(xml_file, "xml")

        authors = self.get_authors(xml_file)

        sentences = xml_file.findAll("s", recursive=True)
        for sentence in sentences:
            curr_sentence = Sentence(sentence["n"], authors)
            for word in sentence.findAll(["w", "c"], recursive=True):
                if word.name == "c":
                    curr_sentence.add_token(
                        Token(word=word.text, is_punctuation=True, is_multiword=word.parent.name == "mw",
                              c5=word["c5"]))
                else:
                    curr_sentence.add_token(
                        Token(word=word.text, is_punctuation=False, is_multiword=word.parent.name == "mw",
                              hw=word["hw"], c5=word["c5"],
                              pos=word["pos"]))
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

    def get_authors(self, xml_file):
        """
        Extracts authors of an XML document.
        :param xml_file: Input file
        :return: list of author names
        """
        return list(map(lambda x: x.text, list(xml_file.find("sourceDesc").findAll("author"))))


# Implement a "Classify" class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class). Make sure that the class contains the relevant fields for
# classification, and the methods in order to complete the tasks:


class Classify:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.chunks = dict()

        self.corpus.init_chunks()
        self.init_classifier()

        male_count, female_count = 0, 0

        for chunk in self.chunks.values():
            if chunk.gender == Gender.female:
                female_count += 1
            else:
                male_count += 1

        print("Female: {}".format(female_count))
        print("Male: {}".format(male_count))

    def init_classifier(self):
        # Filtering out all chunks which have unknown gender
        for chunk in filter(lambda x: x.gender != Gender.unknown, self.corpus.chunks):
            self.chunks[chunk.idx] = chunk


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    corpus = Corpus()

    ctr = 0
    for file in listdir(xml_dir):
        corpus.add_xml_file_to_corpus(path.join(xml_dir, file))
        ctr += 1

        if ctr == 50:
            break
    # corpus.add_xml_file_to_corpus("bnc/A08.xml")

    classifier = Classify(corpus)

    # Implement here your program:
    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    # 2. Create a classification object based on the class implemented above.
    # 3. Classify the chunks of text from the corpus as described in the instructions.
    # 4. Print onto the output file the results from the second task in the wanted format.
