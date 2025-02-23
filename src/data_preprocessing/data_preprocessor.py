import json
import os
import re
from glob import glob
from string import punctuation
from typing import Union

import pandas as pd
import pylangacq
from tqdm import tqdm

from src.data_preprocessing.constants import (IGNORE_POS, IGNORE_TOKENS,
                                              NEOLOGISMS, PUNCT_REPLACEMENTS)
from src.ud_annotation.ud_annotator import UDAnnotator


class DataPreprocessor:
    def __init__(
        self,
        patient_id: str = "PAR",
        allowed_labels: list[str] = ("PossibleAD", "ProbableAD", "MCI", "Control"),
    ):
        self.patient_id = patient_id
        self.ignore_tokens = IGNORE_TOKENS
        self.ignore_pos = IGNORE_POS
        self.punct_replacements = PUNCT_REPLACEMENTS
        self.neologisms = NEOLOGISMS
        self.allowed_labels = allowed_labels
        self.ud_annotator = UDAnnotator()

    def process_dataset(
        self, dataset_path: str, output_path: str, return_output: bool = False
    ):
        data = []
        files_to_process = glob(os.path.join(dataset_path, "*", "*.cha"))

        for file in tqdm(files_to_process):
            participant_data = self.process_file(file)
            if participant_data:
                data.append(participant_data)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        if return_output:
            return data

    def process_file(self, file_path: str) -> Union[dict[str, str], None]:
        full_speech = ""
        on = 0
        co = 0
        data = pylangacq.read_chat(file_path)
        headers = data.headers()[0]
        diagnosis = headers["Participants"]["PAR"]["group"]

        if diagnosis not in self.allowed_labels:
            return None

        if diagnosis in ["PossibleAD", "ProbableAD"]:
            diagnosis = "AD"

        id_ = os.path.basename(file_path.split(".")[0])
        utterances = data.utterances()
        for utterance in utterances:
            if utterance.participant == self.patient_id:
                text, on_sentence, co_sentence = self.get_text(utterance.tokens)
                on += on_sentence
                co += co_sentence
                full_speech = " ".join([full_speech, text])
                # Change several spaces to only one
                full_speech = re.sub(r"\s+", " ", full_speech)
                full_speech = full_speech.strip()
        return {
            "id": id_,
            "diagnosis": diagnosis,
            "speech": full_speech,
            "annotation": self.ud_annotator.annotate_text(full_speech),
            "on": on / len(utterances),
            "co": co / len(utterances),
        }

    def get_unique_pos(
        self, file_path: str, output_json: str, return_output: bool = False
    ):
        pos = {}
        files = glob(os.path.join(file_path, "*", "*.cha"))
        for file in tqdm(files):
            data = pylangacq.read_chat(file)
            id_ = os.path.basename(file.split(".")[0])
            for utterance in data.utterances():
                if utterance.participant == self.patient_id:
                    for token in utterance.tokens:
                        if token.pos not in pos:
                            pos[token.pos] = [[token.word], [id_]]
                        elif token.word not in pos[token.pos][0]:
                            pos[token.pos][0].append(token.word)
                            pos[token.pos][1].append(id_)
        with open(output_json, "w") as json_file:
            json.dump(pos, json_file, indent=4)

        if return_output:
            return pos

    def get_text(self, tokens):
        full_speech = ""
        on = 0
        co = 0
        for idx, token in enumerate(tokens):
            if token.word not in self.ignore_tokens:
                if token.word in self.ignore_tokens or token.pos in self.ignore_pos:
                    if token.pos == "on":
                        on += 1
                    if token.pos == "co":
                        co += 1
                    continue
                if "_" in token.word:
                    words = [full_speech]
                    words.extend(token.word.split("_"))
                    full_speech = " ".join(words)
                # study more tokens of uncomprehensible speech and change them to one tag as well
                elif token.pos == self.neologisms:
                    full_speech = " ".join([full_speech, "NEO"])
                elif token.pos in self.punct_replacements:
                    full_speech = " ".join(
                        [full_speech, self.punct_replacements[token.pos]]
                    )
                else:
                    full_speech = " ".join([full_speech, token.word])

        # If the punctuation is only what left from a sentence, filter it
        if (
            len(
                full_speech.translate(str.maketrans("", "", punctuation)).replace(
                    " ", ""
                )
            )
            == 0
        ):
            full_speech = ""
        if full_speech.startswith(","):
            full_speech = full_speech[1:]
        if full_speech.strip() == "my .":
            full_speech = ""
        # Change several spaces to only one
        full_speech = re.sub(r"\s+", " ", full_speech)
        return full_speech, on, co

    @staticmethod
    def clean_speech_line(text: str) -> str:
        # Remove timestamps like 39140_42140
        text = re.sub(r"\d+_\d+", "", text)
        # Remove symbols like &-uh, &+flow
        text = re.sub(r"&[-+]", "", text)
        # Step 1: Remove content within < > and keep the text inside
        text = re.sub(r"<([^>]+)>", r"\1", text)
        # Step 2: Remove [+ exc] and similar patterns within [ ]
        text = re.sub(r"\[[^\]]+\]", "", text)
        # Step 3: Remove [//]
        text = re.sub(r"\[//\]", "", text)
        # Remove parentheses but keep the content inside
        text = re.sub(r"\(([^)]+)\)", r"\1", text)
        # Remove leading "+" or "+" followed by spaces (e.g., "+<" or "+ text")
        text = re.sub(r"^\+|\+\s*", "", text)
        # Remove any remaining non-alphabetic characters (except spaces, periods, and apostrophes)
        text = re.sub(r"[^a-zA-Z\s.\']", "", text)
        # Remove extra spaces
        text = " ".join(text.split())
        return text.strip()
