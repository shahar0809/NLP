from sys import argv
from typing import List

import numpy as np
from collections import Counter


class Node:
    def __init__(self, symbol: str = "", probability: float = 0, child1=None, child2=None):
        self.symbol = symbol
        self.probability = probability
        self.child1 = child1
        self.child2 = child2


class Rule:
    def __init__(self, probability: float, head: str, derived: str):
        self.probability = probability
        self.head = head
        self.derived = derived.split(" ")

    @staticmethod
    def parse_rule(content: str):
        params = content.split(" ", 1)
        prob, rule = float(params[0]), params[1]

        rule_params = rule.split(" -> ")
        return Rule(prob, rule_params[0], rule_params[1])

    def __eq__(self, other):
        return self.head == other.head and self.derived == other.derived


class Grammar:
    def __init__(self):
        self.rules = list()
        self.unique_tags = None

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def search_rule(self, derivation):
        return filter(lambda rule: rule.derived == derivation, self.rules)

    def generate_tag_ids(self):
        tags = Counter(map(lambda rule: rule.head, self.rules)).keys()
        self.unique_tags = {tag: idx for idx, tag in enumerate(tags)}

    def get_amount_of_tags(self) -> int:
        return len(Counter(map(lambda rule: rule.head, self.rules)))


def parse_cfg(cfg_file: str) -> Grammar:
    """
    Parses a CFG into rules.
    :param cfg_file: The path of the CFG file.
    :return: List of all the rules
    """
    grammar = Grammar()
    with open(cfg_file) as cfg:
        for rule in cfg.read().splitlines():
            grammar.add_rule(Rule.parse_rule(rule))
    return grammar


class Parser:
    """
    Parser that implements the CYK algorithm.
    """

    def __init__(self, grammar: Grammar, sentence: List[str]):
        self.grammar = grammar
        self.sentence = sentence
        self.n = len(sentence)
        self.grammar.generate_tag_ids()
        self.v_size = self.grammar.get_amount_of_tags()

        self.backtrack = [[[] for i in range(self.n + 1)] for j in range(self.n)]

    def generate_parsing_tree(self):
        for j in range(1, self.n + 1):
            for rule in self.grammar.search_rule([self.sentence[j - 1]]):
                self.backtrack[j - 1][j] += [Node(rule.head, rule.probability, Node(self.sentence[j - 1]))]

            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    for B in self.backtrack[i][k]:
                        for C in self.backtrack[k][j]:
                            for rule_A in self.grammar.search_rule([B.symbol, C.symbol]):
                                # Calculate probability according to the algorithm
                                prob = rule_A.probability * B.probability * C.probability
                                self.backtrack[i][j] += [Node(rule_A.head, prob, B, C)]

    def is_in_grammar(self):
        is_true = False
        for node in self.backtrack[0][self.n]:
            is_true = is_true or node.symbol == "S"
        return is_true

    def get_tree_head(self):
        max_node = None
        max_prob = 0
        for node in self.backtrack[0][self.n]:
            if node.symbol == "S" and node.probability > max_prob:
                max_node = node
                max_prob = node.probability
        return max_node

    def draw_parsing_tree(self, output_file):
        output_file.write("Sentence: {}\n".format(" ".join(self.sentence)))
        tree_head = self.get_tree_head()

        if not self.is_in_grammar():
            output_file.write("*** This sentence is not a member of the language generated by the grammar ***\n")
        else:
            output_file.write("Parsing:\n")
            self.recursive_draw(output_file, tree_head, 0)
            output_file.write("Log probability: {}\n".format(np.log(tree_head.probability)))
        output_file.write("\n")

    def recursive_draw(self, output_file, node: Node, indent_size: int):
        output_file.write("{}{}".format(indent_size * "\t", node.symbol))

        # Normal rules, with two non-terminals
        if not (node.child1.symbol.islower() or node.child2.symbol.islower()):
            output_file.write("\n")
            self.recursive_draw(output_file, node.child1, indent_size + 1)
            self.recursive_draw(output_file, node.child2, indent_size + 1)

        # Word rules
        elif node.child1.symbol.islower():
            output_file.write(" > {}\n".format(node.child1.symbol))
        elif node.child2.symbol.islower():
            output_file.write(" > {}\n".format(node.child2.symbol))


def main(input_grammar, input_sentences, output_trees):
    tests = open(input_sentences, "r")
    test_content = tests.read()
    output_file = open(output_trees, "w")

    cfg_grammar = parse_cfg(input_grammar)
    for sentence in test_content.splitlines():
        parser = Parser(cfg_grammar, sentence.split(" "))
        parser.generate_parsing_tree()

        parser.draw_parsing_tree(output_file)

    tests.close()
    output_file.close()


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3])