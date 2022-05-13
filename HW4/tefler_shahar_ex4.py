import re
from collections import Counter, deque
from os import listdir, path
from sys import argv
from typing import List, Tuple, Callable
from xml import etree
from random import uniform, choices

from numpy import log2
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
    def __init__(self, sentence_id: str = "0", is_title=False, tokens: list = None):
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

    def get_token_index(self, word: str):
        for token_idx, token in enumerate(self.tokens):
            if token.word == word.strip().lower():
                return token_idx
        return -1


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
    abbreviations = ["U.S.", "M.A.", "v.", ".com"]

    paragraph_delimiter = "\n"
    sentence_delimiter = "."
    token_delimiter = " "
    title_delimiter = "="

    def __init__(self, sentences: list = None):
        self.tokens_counter = Counter()
        self.sentences = sentences
        self.tweets = list()
        if sentences is None:
            self.sentences = list()
        self.files = list()
        self.max_sentence_length = 0

    def __len__(self) -> int:
        return sum(len(sentence.tokens) for sentence in self.sentences)

    def __str__(self) -> str:
        corpus_str = str()
        for sentence in self.sentences:
            tokens_strings = list()
            for token in sentence.tokens:
                tokens_strings.append(str(token))
            corpus_str += " ".join(tokens_strings) + self.paragraph_delimiter
        return corpus_str

    def add_token(self, sentence: Sentence, token: Token) -> None:
        sentence.add_token(token)
        self.tokens_counter[token.word.lower()] += 1

    def add_xml_file_to_corpus(self, file_name: str) -> None:
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
                    self.add_token(curr_sentence, curr_token)

            if len(curr_sentence.tokens) != 0:
                self.sentences.append(curr_sentence)
        self.max_sentence_length = max(self.max_sentence_length, max(map(len, self.sentences)))

    def add_text_file_to_corpus(self, file_name: str) -> None:
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
                self.sentences.extend(self.tokenize(curr_paragraph))

    def add_song_file_to_corpus(self, file_name: str) -> None:
        self.files.append(file_name)
        text_file = open(file_name, "r", encoding="utf-8")
        text_file_content = text_file.read()

        # Looping over all paragraphs
        for curr_paragraph in text_file_content.split(self.paragraph_delimiter):
            if not self.is_empty(curr_paragraph):
                self.sentences.extend(self.tokenize(curr_paragraph, True))

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

    def tokenize(self, text: str, is_song: bool = False) -> List[Sentence]:
        sentences = list()

        if is_song:
            sentences_text = [text]
        else:
            sentences_text = self.split_to_sentences(text, is_song)

        for sentence_id, curr_sentence in enumerate(
                filter(lambda x: len(x) != 0, sentences_text)):
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

    @staticmethod
    def split_to_sentences(content: str, is_song: bool):
        """
        Splits a segment of text into sentences, while considering abbreviations from a pre-defined list.
        :param is_song: Is the file we're adding a song file
        :param content: The text segment
        :return: list of sentences
        """

        if not is_song:
            regex_pattern = r"(?<=[\)*?!.:;-])"
        else:
            regex_pattern = r"(?<=[?!.;])"
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

        self.ngram = ' '.join(token.word.strip() for token in tokens)

    def __str__(self):
        return self.ngram


class NGramModel:
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

        product = 0
        for token_index in range(self.n - 1, len(phrase.tokens)):
            try:
                product += log2(self.get_token_probability(phrase.tokens[token_index],
                                                           phrase.tokens[token_index - self.n + 1: token_index]))
            except ValueError:
                product += 0
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

    def no_smoothing_probability(self, token: Token, phrases: list):
        try:
            return self.get_count(phrases + [token]) / self.get_count(phrases)
        except ZeroDivisionError:
            return 0

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
        return ' '.join(token.word.strip() for token in ngram)

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

    def random_sentence_length(self):
        return int(uniform(1, self.corpus.max_sentence_length))

    def generate_next_word(self, sentence: Sentence):
        context = sentence.tokens[-(self.n - 1):]
        population = list()
        probabilities = list()

        for token in self.vocabulary:
            population.append(token)
            probabilities.append(self.no_smoothing_probability(Token(word=token), context))

        return Token(choices(population, probabilities)[0])

    def generate_random_sentence(self):
        sentence_length = self.random_sentence_length()
        sentence = Sentence(tokens=[Token(word="<b>")])

        for idx in range(sentence_length):
            next_token = self.generate_next_word(sentence)
            sentence.add_token(next_token)

            if next_token.word == "<e>":
                return sentence
        return sentence


class UnigramModel(NGramModel):
    def __init__(self, corpus: Corpus, smoothing_type: str):
        super().__init__(1, corpus, smoothing_type)
        self.ngrams = self.vocabulary

        # Calculating frequencies dictionary
        corpus_len = float(len(self.corpus))
        self.population = list(self.ngrams.keys())
        self.probabilities = list(map(lambda x: x / corpus_len, self.ngrams.values()))

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

    def generate_next_word(self, sentence: Sentence):
        return Token(choices(self.population, self.probabilities)[0])


class BigramModel(NGramModel):
    def __init__(self, corpus: Corpus, smoothing_type: str):
        super().__init__(2, corpus, smoothing_type)
        self.unigram = UnigramModel(corpus, smoothing_type)
        self.build_ngrams()

    def linear_interpolation(self, token, phrases):
        weights = self.calc_weights()
        return weights[0] * self.unigram.get_token_probability(token, list()) + \
               weights[1] * self.no_smoothing_probability(token, phrases[-1:])


class TrigramModel(NGramModel):
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

    def generate_random_sentence(self):
        sentence_length = self.random_sentence_length()
        sentence = Sentence(tokens=[Token(word="<b>")])
        sentence.add_token(self.bigram.generate_next_word(sentence))

        for idx in range(sentence_length - 1):
            next_token = self.generate_next_word(sentence)
            sentence.add_token(next_token)

            if next_token.word == "<e>":
                return sentence
        return sentence


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


def lyrics_to_replace() -> List[str]:
    return ["baby", "doing", "doing", "at", "you", "got", "plans", "say", "trap", "sipping", "sip", "robe", "drip",
            "good", "look", "alone", "woo", "house", "house", "pool", "warm", "shaved", "newborn", "dancing", "east",
            "mansion", "games", "say", "arms", "door", "door", "door", "door", "way", "tonight", "tell", "sweet",
            "sweet", "tight", "tight", "bite", "ah", "unless", "unless", "smoke", "smoke", "got", "haze", "hungry",
            "got", "woo", "keep", "waiting", "making", "shamone", "kissing", "bathtub", "jump", "games", "say", "if",
            "if", "tryna", "arms", "door", "door", "door", "door", "way", "tonight", "tell", "girl", "la", "need", "la",
            "see", "la", "give", "ah", "door", "door", "door", "door", "hoping", "feel", "tonight", "coming", "woo",
            "woo", "la", "la", "coming", "for", "waiting", "adore", "la", "la"]


def format_new_song(song_filename: str, file_directory: str, model: KeyedVectors, output,
                    old_words: List[str]) -> None:
    song_corpus = Corpus()
    song_corpus.add_song_file_to_corpus(song_filename)

    general_corpus = Corpus()
    for file in listdir(file_directory):
        general_corpus.add_xml_file_to_corpus(path.join(file_directory, file))

    for sentence_idx, sentence in enumerate(song_corpus.sentences):
        old_word = old_words[sentence_idx]
        top_similar_words = model.most_similar(old_word, topn=10)

        token_idx = sentence.get_token_index(old_word)
        # Finding word with most trigrams count
        most_similar_word = get_most_similar_word(old_word, sentence, top_similar_words, general_corpus)
        # Replace with most similar word
        sentence.tokens[token_idx] = most_similar_word

    output.write("=== New Hit ===\n\n\n")
    output.write(str(song_corpus))


def get_most_similar_word(old_word: str, sentence: Sentence, similar_words: List[Tuple[str, float]],
                          corpus: Corpus) -> str:
    ngram_counter = Counter()
    trigram_model = TrigramModel(corpus, smoothing_type="Linear interpolation")

    token_idx = sentence.get_token_index(old_word)
    for similar_word, vector in similar_words:
        # words = list()
        # trigrams for sentences with at least three tokens
        if len(sentence.tokens) >= 3:
            if token_idx == len(sentence.tokens) - 1:
                words = [sentence.tokens[token_idx - 2], sentence.tokens[token_idx - 1], Token(similar_word)]
            elif token_idx == 0:
                words = [Token(similar_word), sentence.tokens[token_idx + 1], sentence.tokens[token_idx + 2]]
            else:
                words = [sentence.tokens[token_idx - 1], similar_word, sentence.tokens[token_idx + 1]]
        # unigrams for sentences with only one token
        elif len(sentence.tokens) == 1:
            words = [corpus.tokens_counter[similar_word]]
        # bigrams for sentences with only two token
        else:
            if token_idx == 0:
                words = [Token(similar_word), sentence.tokens[token_idx + 1]]
            else:
                words = [sentence.tokens[token_idx - 1], Token(similar_word)]
        ngram_counter[similar_word] = trigram_model.get_count(words)

    # Return the similar words with the most ngrams counts
    return max(ngram_counter, key=ngram_counter.get)


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
    format_new_song(lyrics_file, xml_dir, pre_trained_model, output_file, lyrics_to_replace())
    format_tweets(tweets_file, pre_trained_model)

    corpus = Corpus()
    # for file in listdir(xml_dir):
    #     corpus.add_xml_file_to_corpus(path.join(xml_dir, file))
