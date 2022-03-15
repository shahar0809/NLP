from sys import argv
from os import path


class Token:
    def __init__(self, token_id, sentence_id: int, c5: str, word: str, pos: str):
        self.token_id = token_id
        self.sentence_id = sentence_id
        self.c5 = c5
        self.word = word
        self.pos = pos


class Sentence:
    def __init__(self, tokens: list=None):
        self.tokens = tokens

    def add_token(self, token: Token):
        self.tokens.append(token)

    def get_token(self, token_id: int):
        return filter(lambda token:token.token_id == token_id, self.tokens)


class Corpus:
    symbols = [",", ":", "(", ")"]
    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "

    def __init__(self, sentences: list):
        self.sentences = sentences
        self.files = list()

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        self.files.append(file_name)

        # Start parsing XML file



    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """

        def is_paragraph(paragraph: str):
            return "==" not in paragraph != "\n"

        def preprocess(text: str):
            for symbol in self.symbols:
                text = text.replace(symbol, "")
            return text.replace("-", " ")

        self.files.append(file_name)
        text_file = open(file_name, "r")
        text_file_content = preprocess(text_file.read())

        # Split to paragraphs
        for paragraph_id, curr_paragraph in enumerate(text_file_content.split(self.paragraph_delimiter)):
            if is_paragraph(curr_paragraph):
                sentence = Sentence()
                # Split to sentences
                for sentence_id, curr_sentence in enumerate(curr_paragraph.split(self.sentence_delimiter)):
                    # Split to tokens
                    for token_id, curr_token in enumerate(curr_sentence.split(self.token_delimiter)):
                        token = Token(token_id, sentence_id, "", curr_token, "")
                        sentence.add_token(token)
                    self.sentences.append(sentence)


    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """
        return

class File:
    def __init__(self, file_name: str):
        self.file_name = file_name

class XML_File(File):
    def parse_file(self):
        pass

class TextFile(File):
    def parse_file(self):
        pass

    def unparse_file(self):


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus (not a zip file)
    wiki_dir = argv[2]  # directory containing text files from Wikipedia (not a zip file)
    output_file = argv[3]

    # Implement your program here after implementing the classes and their methods above.
    # Your program should do the following (using the methods that need to be implemented):
    # 1. Create a corpus object (after implementing it).
    # 2. Read the XML files from the XML directory to the corpus.
    # 3. Read the text files from the Wikipedia directory to the corpus.
    # 4. Write the content of the whole corpus to the output file.
