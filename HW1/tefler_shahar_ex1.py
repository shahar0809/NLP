from sys import argv
from os import path
from os import listdir


class Token:
    def __init__(self, token_id, sentence_id: int, word: str, is_title, c5: str = "", pos: str = ""):
        self.token_id = token_id
        self.sentence_id = sentence_id
        self.c5 = c5
        self.word = word
        self.pos = pos
        self.is_title = is_title

    def __str__(self):
        return self.word


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
    symbols = [",", ":", "(", ")"]
    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "

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

        # Start parsing XML file

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """

        self.files.append(file_name)
        text_file = open(file_name, "r", encoding="utf-8")
        text_file_content = self.preprocess(text_file.read())

        # Looping over all paragraphs
        for paragraph_id, curr_paragraph in enumerate(text_file_content.split(self.paragraph_delimiter)):
            if not self.is_empty(curr_paragraph):
                # Looping over all sentences in paragraph
                for sentence_id, curr_sentence in enumerate(curr_paragraph.split(self.sentence_delimiter)):
                    sentence = Sentence()

                    # Looping over all tokens in sentence
                    for token_id, curr_token in enumerate(curr_sentence.split(self.token_delimiter)):
                        sentence.add_token(Token(token_id, sentence_id, curr_token, self.is_title(curr_sentence)))
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
            file.write((" ".join(tokens_strings) + self.paragraph_delimiter).encode())
        file.close()

    def preprocess(self, text: str):
        for symbol in self.symbols:
            text = text.replace(symbol, "")
        return text.replace("-", " ")

    @staticmethod
    def is_title(paragraph: str):
        return "==" in paragraph

    @staticmethod
    def is_empty(content: str):
        return content == "\n" or content == ""


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
        pass


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

    # Implement your program here after implementing the classes and their methods above.
    # Your program should do the following (using the methods that need to be implemented):
    # 1. Create a corpus object (after implementing it).
    # 2. Read the XML files from the XML directory to the corpus.
    # 3. Read the text files from the Wikipedia directory to the corpus.
    # 4. Write the content of the whole corpus to the output file.
