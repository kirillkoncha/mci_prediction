import ast
import math
from collections import defaultdict

import conllu
import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from pycpidr import depid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.features_extraction.constants import (
    LOW_SPECIFICITY_SENTENCES,
    PATH_TO_WORD_CLUSTERS,
)

nltk.download("punkt")
nltk.download("punkt_tab")


class FeaturesExtractor:
    def __init__(
        self,
        sentence_embeddings_model: str | None = None,
        surprisal_model: str | None = None,
    ):
        self.low_specificity_sentences: list[str] = LOW_SPECIFICITY_SENTENCES
        self.sentence_embeddings_model: str | None = sentence_embeddings_model
        self.surprisal_model: str | None = surprisal_model
        self.surprisal_tokenizer: transformers.PreTrainedTokenizerFast | None = None
        self.cluster_words: pd.DataFrame = pd.read_csv(PATH_TO_WORD_CLUSTERS)
        self.cluster_words["words"]: pd.core.series.Series = self.cluster_words[
            "words"
        ].apply(ast.literal_eval)
        self.sid_icus: list[str | None] = []

        for word_list in self.cluster_words["words"].to_list():
            self.sid_icus.extend(word_list)

        nltk.download("stopwords")
        from nltk.corpus import stopwords

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

    def extract_sentence_surprisal(self, text: str, model: str = "gpt2") -> float:
        """
        Extracts mean sentence surprisal of a text

        Args:
            text (str): Text speech
            model (str): HuggingFace model path

        Returns:
            float: Mean sentence suprisal of a text
        """
        if self.surprisal_model is None:
            self.surprisal_model = GPT2LMHeadModel.from_pretrained(model)
            self.surprisal_model.eval()
        if self.surprisal_tokenizer is None:
            self.surprisal_tokenizer = GPT2TokenizerFast.from_pretrained(model)

        inputs = self.surprisal_tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = self.surprisal_model(input_ids, labels=input_ids)
            loss_per_token = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            )

        surprisal_per_token = loss_per_token / torch.log(torch.tensor(2.0))
        mean_surprisal = surprisal_per_token.mean().item()

        return mean_surprisal

    def extract_mlu(self, conllu_annotation: str) -> float:
        """
        Extracts mean number of words in sentences

        Args:
            conllu_annotation (str): Conllu anotation of the whole speech
        Returns:
            float: Mean number of words in a sentence
        """
        text = conllu.parse(conllu_annotation)
        token_counter = 0

        for sentence in text:
            token_counter += self._get_sentence_length(sentence)

        if token_counter == 0:
            return 0.0

        return token_counter / len(text)

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

    def extract_words_per_clause(self, conllu_annotation: str) -> float:
        """
        Calculate the mean number of words (excluding punctuation) per clause in English CONLLU text.

        Args:
            conllu_annotation (str): Conllu annotation of the whole speech

        Returns:
            float: mean number of words per clause in a text
        """
        sentences = conllu.parse(conllu_annotation)
        total_content_words = 0
        total_clauses = 0

        for sentence in sentences:
            content_words_in_sentence = 0
            clause_heads = set()

            for token in sentence:
                if token.get("upos") == "PUNCT":
                    continue

                content_words_in_sentence += 1

                # Clause heads in English
                if token["deprel"] == "root":
                    clause_heads.add(token["id"])
                elif token["deprel"] in [
                    "ccomp",
                    "xcomp",
                    "advcl",
                    "acl",
                    "parataxis",
                    "csubj",
                    "csubj:pass",
                    "acl:relcl",
                ]:
                    clause_heads.add(token["head"])
                elif token["deprel"] == "mark" or (
                    token["deprel"] == "aux" and token.get("lemma") == "to"
                ):
                    clause_heads.add(token["head"])

            # Handle coordination (e.g., "She ran and he left")
            for token in sentence:
                if token["deprel"] == "conj" and token["head"] in clause_heads:
                    clause_heads.add(token["id"])

            # Fallback to root token if no clause heads found
            if not clause_heads:
                root_token = next(
                    (token for token in sentence if token["deprel"] == "root"), None
                )
                if root_token:
                    clause_heads.add(root_token["id"])
                else:
                    clause_heads.add(1)  # Emergency fallback

            total_content_words += content_words_in_sentence
            total_clauses += len(clause_heads) if clause_heads else 1

        if total_clauses == 0:
            return 0.0

        return total_content_words / total_clauses

    def extract_frazier_score(self, conllu_text):
        """
        Extracts Frazier score which quantifies syntactic complexity by scoring nodes based on their 
        position in dependency tree. Score increases for root nodes and leftmost children. According to
        Roark et al. (2007).

        Args:
            conllu_text (str): Conllu annotation of the whole speech

        Returns:
            float: Overall mean Frazier score across all sentences
        """
        trees = conllu.parse(conllu_text)
        results = []

        for tree in trees:
            scores = []
            token_dict = {token["id"]: token for token in tree}

            for token in tree:
                score = 0
                node = token

                while node:
                    head_id = node["head"]
                    if head_id == 0:  # Root reached
                        score += 1.5
                        break

                    parent = token_dict.get(head_id)
                    if parent:
                        siblings = [t for t in tree if t["head"] == head_id]
                        if (
                                siblings and siblings[0]["id"] == node["id"]
                        ):  # Leftmost child
                            score += 1
                        else:
                            break  # Stop at first non-leftmost child

                    node = parent

                scores.append(score)

            total_score = sum(scores)
            mean_score = total_score / len(scores)

            results.append(mean_score)

        overall_mean_score = sum(results) / len(results) if results else 0

        return overall_mean_score

    def extract_maas_index(self, conllu_annotation: str) -> float:
        """
        Computes the Maas lexical diversity index, which considers both the number of unique words 
        and total words in a text. Lower values indicate higher lexical diversity.
    
        Args:
            conllu_annotation (str): Conllu annotation of the whole speech
    
        Returns:
            float: Maas lexical diversity index of the text
        """
        text = conllu.parse(conllu_annotation)

        words = [
            token["form"].lower()
            for sentence in text
            for token in sentence
            if token["upos"] != "PUNCT"
        ]

        total_words = len(words)
        unique_words = len(set(words))

        if total_words == 0 or unique_words == 0:
            return 0.0

        mi = (math.log(total_words) - math.log(unique_words)) / (
            math.log(total_words) ** 2
        )
        return mi

    def _get_sentence_length(self, sentence: conllu.models.TokenList) -> int:
        """
        Counts how many word tokens are in the sentence. The results are cached

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

    def extract_words_per_clause(self, conllu_text: str) -> float:
        """
        Computes the average number of words per clause in the given CoNLL-U formatted
        text.

        Args:
            conllu_text (str): A string representing the CoNLL-U formatted text.
        Returns:
            float: The computed average number of words per clause as a float
        """
        trees = conllu.parse(conllu_text)
        total_words = 0
        total_clauses = 0

        for tree in trees:
            total_words += self._get_sentence_length(tree)
            total_clauses += sum(
                1
                for token in tree
                if token["deprel"]
                in {
                    "root",
                    "ccomp",
                    "advcl",
                    "acl",
                    "xcomp",
                    "parataxis",
                    "conj",
                    "relcl",
                }
            )

        words_per_clause = total_words / total_clauses if total_clauses > 0 else 0

        return words_per_clause

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
