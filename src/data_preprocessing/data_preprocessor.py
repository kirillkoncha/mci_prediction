import json
import os
import re
from collections import Counter
from glob import glob
from string import punctuation
from typing import Union

import pandas as pd
import pylangacq
from tqdm import tqdm

from src.data_preprocessing.constants import (
    IGNORE_POS,
    IGNORE_TOKENS,
    NEOLOGISMS,
    PUNCT_REPLACEMENTS,
)
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
        self, dataset_path: str, output_path: str, mmse_xlsx_path: str, return_output: bool = False, match_mmse: bool = False) -> Union[pd.DataFrame, None]:
        """
        Process whole dataset. Dataset should have the following structure:
            root
                controls
                    .cha
                    .cha
                    ...
                dementia
                    .cha
                    .cha
                    ...

        Args:
            dataset_path (str): Path to folder with dataset
            output_path (str): Path to save results
            mmse_xlsx_path (str): Path to MMSE scores .xlsx file from DementiaBank
            return_output (bool): Returns results to a variable if is True
            match_mmse (bool): If True, matches MMSE scores to participants' data. Defaults to False.

        Returns:
            pd.DataFrame | None: Returns pd.DataFrame if return_output is set to true
            or None otherwise
        """
        data = []
        files_to_process = glob(os.path.join(dataset_path, "*", "*.cha"))

        for file in tqdm(files_to_process):
            participant_data = self.process_file(file)
            if participant_data:
                data.append(participant_data)

        df = pd.DataFrame(data)

        if match_mmse:
            df = self.match_mmse(df, mmse_xlsx_path)
        df.to_csv(output_path, index=False)
        if return_output:
            return df
        else:
            return None

    def process_file(self, file_path: str) -> Union[dict[str, Union[str, float]], None]:
        """
        Process file with participants data (participants should be Controls, MCI, or AD)

        Args:
            file_path (str): Path to .cha file

        Returns:
            dict[str, str | float] | None: Dictionary with participants data and processed utterances
            or None if participants does not belong to Control, MCI, or AD group
        """
        full_speech = ""
        on = 0
        co = 0
        data = pylangacq.read_chat(file_path)
        headers = data.headers()[0]
        diagnosis = headers["Participants"]["PAR"]["group"]
        total_speaking_time = 0
        pause_regex = re.compile(r"\(\.\)|\(\.\.\)|\(\.\.\.\)")
        pause_counter = {"(.)": 0, "(..)": 0, "(...)": 0}

        if diagnosis not in self.allowed_labels:
            return None

        if "AD" in diagnosis:
            diagnosis = "AD"

        id_ = os.path.basename(file_path.split(".")[0])
        utterances = data.utterances()
        for utterance in utterances:
            if utterance.participant == self.patient_id:
                text_full = utterance.tiers.get(self.patient_id)
                if text_full:
                    matches = pause_regex.findall(text_full)
                    for match_ in matches:
                        pause_counter[f"{match_}"] += 1
                time_marks = utterance.time_marks
                if time_marks:
                    start_time, end_time = time_marks
                    total_speaking_time += end_time - start_time
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
            "speaking time (s)": total_speaking_time / 1000,
            "on": on / len(utterances),
            "co": co / len(utterances),
            "short_pause": pause_counter["(.)"],
            "mid_pause": pause_counter["(..)"],
            "long_pause": pause_counter["(...)"],
        }

    def match_mmse(self, mmse_xlsx_path: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function that matches MMSE scores to participants' data

        Args:
            mmse_xlsx_path (str): Path to MMSE scores .xlsx file from DementiaBank
            df (pd.DataFrame): DataFrame with participants' data

        Returns:
            pd.DataFrame: DataFrame with matched MMSE scores
        """
        def get_mmse(row):
            main_id = row['main_id']
            sub_id = row['sub_id']
            mmse_column = f"mmse{sub_id}"

            # Check if the column exists in the "match" sheet
            if mmse_column in excel_df.columns:
                result = excel_df.loc[excel_df['id'] == main_id, mmse_column]
                return result.values[0] if not result.empty else None
            else:
                return None

        excel_df = pd.read_excel(mmse_xlsx_path, sheet_name='match')
        df['main_id'] = df['id'].apply(lambda x: int(x.split('-')[0]))  # Main ID as integer
        df['sub_id'] = df['id'].apply(lambda x: x.split('-')[1])  # Sub-ID as a string
        df['mmse'] = df.apply(get_mmse, axis=1)
        df.drop(columns=['main_id', 'sub_id'], inplace=True)
        filtered_df = df[df['mmse'].isna()]
        result = df.merge(filtered_df, on=df.columns.tolist(), how='left', indicator=True)
        result = result[result['_merge'] == 'left_only'].drop(columns='_merge')
        return result

    def get_unique_pos(
        self, file_path: str, output_json: str, return_output: bool = False
    ) -> Union[None, dict[str, str]]:
        """
        Function to get unique POS tags and participants who use them

        Args:
            file_path (str): Path to Pitt Corpus
            output_json (str): Path to output file
            return_output (bool): Returns results to a variable if is True

        Returns:
            dict[str, str] | None: Dictionary with POS tags as keys and participants who use them as values
            or None if return_output is set to False
        """
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

    def get_text(
        self, tokens: list[pylangacq.objects.Token]
    ) -> tuple[str, float, float]:
        """
        Function to process single text. Filters out POS and tokens that should be ignored;
        replaces neologisms with word NEO; counts interjections/interactions and onomatopoeia
        rates per text

        Args:
            tokens (list): List of tokens from pylangacq

        Returns:
            tuple[str, float, float]: Preprocessed utterance, interjections/interactions rate, onomatopoeia rate
        """
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
