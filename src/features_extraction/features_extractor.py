import conllu

from pycpidr import depid

from src.features_extraction.constants import (DET_NO,
                                               LOW_SPECIFICITY_SENTENCES,
                                               PID_DEPRELS, SID_NSUBJ_NO)


class FeaturesExtractor:
    def __init__(self):
        self.pid_deprels = PID_DEPRELS

        self.sid_nsubj_no = SID_NSUBJ_NO
        self.det_no = DET_NO
        self.low_specificity_sentences = LOW_SPECIFICITY_SENTENCES

    def extract_tree_depth(self, text: str) -> float:
        text = conllu.parse(text)
        depth = 0

        for sentence in text:
            depth += self._extract_one_tree_depth(sentence)

        return depth/len(text)

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
