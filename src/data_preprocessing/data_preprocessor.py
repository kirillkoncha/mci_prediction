import os
import re

import pylangacq

from src.data_preprocessing.constants import IGNORE_TOKENS


class DataPreprocessor:
    def __init__(self, patient_id: str = "PAR"):
        self.patient_id = patient_id
        self.ignore_tokens = IGNORE_TOKENS

    def process_file(self, file_path: str) -> dict[str, str]:
        full_speech = ""
        data = pylangacq.read_chat(file_path)
        headers = data.headers()[0]
        diagnosis = headers["Participants"]["PAR"]["group"]
        id_ = os.path.basename(file_path.split(".")[0])

        for utterance in data.utterances():
            if utterance.participant == self.patient_id:
                text = self.get_text(utterance.tokens)
                full_speech = " ".join([full_speech, text])
        return {"id": id_, "diagnosis": diagnosis, "speech": full_speech.strip()}

    def get_text(self, tokens):
        full_speech = ""
        for token in tokens:
            if token.word not in self.ignore_tokens:
                if "_" in token.word:
                    words = [full_speech]
                    words.extend(token.word.split("_"))
                    full_speech = " ".join(words)
                # study more tokens of uncomprehensible speech and change them to one tag as well
                elif token.pos == "neo":
                    full_speech = " ".join([full_speech, "NEO"])
                else:
                    full_speech = " ".join([full_speech, token.word])
        return full_speech

    @staticmethod
    def clean_speech_line(text: str) -> str:
        # Remove timestamps like 39140_42140
        text = re.sub(r'\d+_\d+', '', text)
        # Remove symbols like &-uh, &+flow
        text = re.sub(r'&[-+]', '', text)
        # Step 1: Remove content within < > and keep the text inside
        text = re.sub(r'<([^>]+)>', r'\1', text)
        # Step 2: Remove [+ exc] and similar patterns within [ ]
        text = re.sub(r'\[[^\]]+\]', '', text)
        # Step 3: Remove [//]
        text = re.sub(r'\[//\]', '', text)
        # Remove parentheses but keep the content inside
        text = re.sub(r'\(([^)]+)\)', r'\1', text)
        # Remove leading "+" or "+" followed by spaces (e.g., "+<" or "+ text")
        text = re.sub(r'^\+|\+\s*', '', text)
        # Remove any remaining non-alphabetic characters (except spaces, periods, and apostrophes)
        text = re.sub(r'[^a-zA-Z\s.\']', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text.strip()
