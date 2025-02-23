import ast

import conllu
import nltk
import numpy as np
import pandas as pd
from pycpidr import depid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from src.features_extraction.constants import (LOW_SPECIFICITY_SENTENCES,
                                               PATH_TO_WORD_CLUSTERS)

nltk.download("punkt")
nltk.download("punkt_tab")


class FeaturesExtractor:
    def __init__(self, sentence_embeddings_model: str = None):
        self.low_specificity_sentences = LOW_SPECIFICITY_SENTENCES
        self.sentence_embeddings_model = sentence_embeddings_model
        self.cluster_words = pd.read_csv(PATH_TO_WORD_CLUSTERS)
        self.cluster_words["words"] = self.cluster_words["words"].apply(
            ast.literal_eval
        )
        self.sid_icus = []

        for word_list in self.cluster_words["words"].to_list():
            self.sid_icus.extend(word_list)

    def extract_sentence_embeddings(
        self, text: str, model: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> float:
        if self.sentence_embeddings_model is None:
            self.sentence_embeddings_model = SentenceTransformer(model)

        sentences = nltk.sent_tokenize(text)

        if len(sentences) < 2:
            return 0

        embeddings = self.sentence_embeddings_model.encode(sentences)

        distances = cosine_distances(embeddings)

        n = len(sentences)
        upper_tri_indices = np.triu_indices(n, k=1)
        mean_distance = distances[upper_tri_indices].mean()

        return mean_distance

    def extract_tree_depth(self, text: str) -> float:
        text = conllu.parse(text)
        depth = 0

        for sentence in text:
            depth += self._extract_one_tree_depth(sentence)

        return depth / len(text)

    def extract_verbs_with_inflections(self, conllu_annotation: str):
        token_counter = 0
        verbs_number = 0

        text_conllu = conllu.parse(conllu_annotation)
        for sentence in text_conllu:
            token_counter += self._get_sentence_length(sentence)
            for token in sentence:
                if token["form"] != token["lemma"]:
                    verbs_number += 1
        if token_counter == 0:
            return 0.0
        return verbs_number / token_counter

    def extract_nouns_with_determiners(self, conllu_annotation: str) -> float:
        token_counter = 0
        nouns_with_determiners = 0

        text_conllu = conllu.parse(conllu_annotation)
        for sentence in text_conllu:
            token_counter += self._get_sentence_length(sentence)
            for token in sentence:
                if (
                    token["upos"] == "DET"
                    and sentence[token["head"] - 1]["upos"] == "NOUN"
                ):
                    nouns_with_determiners += 1
        if token_counter == 0:
            return 0.0
        return nouns_with_determiners / token_counter

    def extract_sid(self, conllu_annotation: str) -> float:
        sid_score = 0
        token_counter = 0

        text_conllu = conllu.parse(conllu_annotation)
        for sentence in text_conllu:
            token_counter += self._get_sentence_length(sentence)
            for token in sentence:
                if token["lemma"].lower() in self.sid_icus:
                    sid_score += 1
        if token_counter == 0:
            return 0.0
        return sid_score / token_counter

    def extract_pid(self, text: str) -> float:
        """
        Counts Propositional Idea Density according to Sirts et al. (2017)

        Args:
            text (str): Text speech

        Returns:
            float: Propositional Idea Density normalised by token count
        """
        text = conllu.parse(text)
        score = 0
        for sentence in text:
            if not self._filter_sentence_nsubj(sentence):
                continue
            if not self._filter_sentence_specifity(sentence):
                continue
            pid, _, _ = depid(sentence.metadata["text"], is_depid_r=True)
            score += pid
        return score / len(text)

    def _get_sentence_length(self, sentence: conllu.models.TokenList) -> int:
        """
        Counts how many word tokens are in the sentence

        Args:
            sentence (conllu.models.TokenList): Token list of a sentence

        Returns:
            int: Number of word tokens in a sentence
        """
        length = 0
        for token in sentence:
            if token["deprel"] != "punct":
                length += 1
        return length

    def _filter_sentence_nsubj(self, sentence: conllu.models.TokenList) -> bool:
        # TO DO: upgrade it so it won't filter out sentences like "i see the young boy"
        """
        Checks if a sentence contains the subject which lemma is "i" or "you"
        That filter was applied in Sirts et al. (2017)

        Args:
            sentence (conllu.models.TokenList): Token list of a sentence

        Returns:
            bool: True if the sentence does not contain the subject which lemma is "i" or "you"
        """
        for token in sentence:
            if (
                token["deprel"] == "nsubj"
                and token["upos"] == "PRON"
                and token["lemma"].split("'")[0].lower() in ["i", "you"]
                and sentence[token["head"] - 1]["deprel"] == "root"
            ):
                return False
        return True

    def _filter_sentence_specifity(self, sentence: conllu.models.TokenList) -> bool:
        """
        Checks if a sentence is in a low specificity sentence list
        That filter was applied in Sirts et al. (2017)

        Args:
            sentence (conllu.models.TokenList): Token list of a sentence

        Returns:
            bool: True if the sentence is not in a low specificity sentence list
        """
        if sentence.metadata["text"] in self.low_specificity_sentences:
            return False
        return True

    def _extract_one_tree_depth(self, sentence: conllu.models.TokenList) -> int:
        """
        Compute the tree depth of a given sentence. From Taktasheva et al. (2024)

        Args:
            sentence (conllu.models.TokenList): Token list of a sentence

        Returns:
            int: Tree depth of a sentence
        """
        tree = sentence.to_tree()
        depth = 0
        stack = [tree]
        while len(stack):
            curr_node = stack[0]
            stack.pop(0)
            if curr_node.children:
                depth += 1
            for node in range(len(curr_node.children) - 1, -1, -1):
                stack.insert(0, curr_node.children[node])
        return depth
