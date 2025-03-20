import ast

import conllu
import nltk
import numpy as np
import pandas as pd
import spacy
from morphemes import Morphemes
from pycpidr import depid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from src.features_extraction.constants import (
    LOW_SPECIFICITY_SENTENCES,
    PATH_TO_WORD_CLUSTERS,
)

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

        nltk.download("stopwords")
        from nltk.corpus import stopwords

        self.morphemes = Morphemes("./morphemes_data")
        self.stop_words = set(stopwords.words("english"))

    def extract_sentence_embeddings(
        self, text: str, model: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> float:
        """
        Extracts mean pairwise distance between sentence embeddings

        Args:
            text (str): Text speech
            model (str): HuggingFace model path

        Returns:
            float: Mean pairwise distance between embeddings of text sentences
        """
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

    def extract_mlu(self, conllu_annotation: str) -> float:
        text = conllu.parse(conllu_annotation)
        token_counter = 0
        morpheme_counter = 0

        for sentence in text:
            token_counter += self._get_sentence_length(sentence)
            for token in sentence:
                if token["deprel"] != "punct":
                    morpheme_counter += self.morphemes.parse(token["form"])[
                        "morpheme_count"
                    ]

        if token_counter == 0:
            return 0.0

        return morpheme_counter / token_counter

    def extract_stop_words(self, conllu_annotation: str) -> float:
        """
        Extracts number of stopwords per number of tokens

        Args:
            conllu_annotation (str): Conllu anotation of the whole speech

        Returns:
            float: Number of stopwords per number of tokens
        """
        text = conllu.parse(conllu_annotation)
        token_counter = 0
        stop_words = 0

        for sentence in text:
            token_counter += self._get_sentence_length(sentence)
            for token in sentence:
                if token["lemma"] in self.stop_words:
                    stop_words += 1

        if token_counter == 0:
            return 0.0
        return stop_words / token_counter

    def extract_tree_depth(self, text: str) -> float:
        """
        Extracts mean syntactic tree depth of the whole text

        Args:
            text (str): Conllu annotation of the whole speech

        Returns:
            float: Average tree depth of all sentences
        """
        text = conllu.parse(text)
        depth = 0

        for sentence in text:
            depth += self._extract_one_tree_depth(sentence)
        if depth == 0:
            return 0.0
        return depth / len(text)

    def extract_verbs_with_inflections(self, conllu_annotation: str) -> float:
        """
        Extracts number of verbs with inflections normalised by token count

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech

        Returns:
            float: Number of verbs with inflections normalised by token count
        """
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
        """
        Extracts number of nouns with determiners normalised by token count

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech

        Returns:
            float: Number of nouns with determiners normalised by token count
        """
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
        """
        Extracts Semantic Idea Density according to Yancheva and Rudzicz (2016).
        Information Content Units are already extracted via K-Means clustering (see
        src.features_extraction.sid_kmeans)

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech

        Returns:
            float: Semantic Idea Density normalised by token count
        """
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

    def extract_sid_efficiency(self, conllu_annotation: str, time: float) -> float:
        """
        Counts Semantic Idea Efficiency, i.e., SID divided, but divided by speech time (in seconds),
        not number of tokens. According to Fraser et al. (2018)

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech
            time (float): Speaking time in seconds

        Returns:
            float: Semantic Idea Efficiency
        """
        sid_score = 0

        text_conllu = conllu.parse(conllu_annotation)
        for sentence in text_conllu:
            for token in sentence:
                if token["lemma"].lower() in self.sid_icus:
                    sid_score += 1
        if time == 0:
            return 0.0
        return sid_score / time

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
        spacy.load("en_core_web_sm")
        for sentence in text:
            if not self._filter_sentence_nsubj(sentence):
                continue
            if not self._filter_sentence_specifity(sentence):
                continue
            pid, _, _ = depid(sentence.metadata["text"], is_depid_r=True)
            score += pid
        return score / len(text)

    def extract_pid_efficiency(self, conllu_annotation: str, time: float) -> float:
        """
        Counts Propositional Idea Efficiency, i.e., PID, but divided by speech time (in seconds),
        not number of tokens. According to Fraser et al. (2018)

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech
            time (float): Speaking time in seconds

        Returns:
            float: Propositional Idea Efficiency
        """
        text_conllu = conllu.parse(conllu_annotation)
        pid_score = 0
        for sentence in text_conllu:
            if not self._filter_sentence_nsubj(sentence):
                continue
            if not self._filter_sentence_specifity(sentence):
                continue
            _, _, dependencies = depid(sentence.metadata["text"], is_depid_r=True)
            pid_score += len(dependencies)

        if time == 0:
            return 0.0
        return pid_score / time

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
