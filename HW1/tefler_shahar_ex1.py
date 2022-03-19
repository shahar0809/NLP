from sys import argv
from os import path
from os import listdir

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
    def __init__(self, is_title=False, tokens: list = None):
        self.is_title = is_title
        self.tokens = tokens
        if tokens is None:
            self.tokens = list()

    def add_token(self, token: Token):
        self.tokens.append(token)

    def get_token(self, token_id: int):
        return filter(lambda token: token.token_id == token_id, self.tokens)


class Corpus:
    sentence_split_delimiters = ["?", "!", ";", ":", "-"]
    abbreviations = ["U.S.", "M.A.", "v."]

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
        xml_file = BeautifulSoup(xml_file, "xml")

        sentences = xml_file.findAll("s", recursive=True)
        for sentence in sentences:
            curr_sentence = Sentence()
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
                for curr_sentence in self.split_to_sentences(curr_paragraph):
                    if not self.is_empty(curr_sentence):
                        sentence = Sentence(is_title=self.title_delimiter in curr_sentence)

                        # Looping over all tokens in sentence
                        for curr_token in curr_sentence.split(self.token_delimiter):
                            curr_token = curr_token.replace(self.title_delimiter, "")
                            if not self.is_empty(curr_token):
                                sentence.add_token(Token(curr_token, self.is_title(curr_sentence), False))
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

    @staticmethod
    def preprocess(text: str):
        pass

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

    def split_to_tokens(self, content):
        pass

    @staticmethod
    def is_title(paragraph: str):
        return "==" in paragraph

    @staticmethod
    def is_empty(content: str):
        return re.compile("[\\n\\r]+").match(content) or content == ""


def main():
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus (not a zip file)
    wiki_dir = argv[2]  # directory containing text files from Wikipedia (not a zip file)
    output_file = argv[3]

    corpus = Corpus()
    for xml_file in listdir(xml_dir):
        corpus.add_xml_file_to_corpus(path.join(xml_dir, xml_file))
    for text_file in listdir(wiki_dir):
        corpus.add_text_file_to_corpus(path.join(wiki_dir, text_file))

    corpus.create_text_file(output_file)


if __name__ == '__main__':
    main()
