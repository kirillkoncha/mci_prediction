import os
import re

import pylangacq


class DataPreprocessor:
    def __init__(self, patient_id: str = "PAR"):
        self.patient_id = patient_id

    def process_file(self, file_path: str) -> dict[str, str]:
        full_speech = ""
        data = pylangacq.read_chat(file_path)
        headers = data.headers()[0]
        diagnosis = headers["Participants"]["PAR"]["group"]
        id_ = os.path.basename(file_path.split(".")[0])

        for utterance in data.utterances():
            if utterance.participant == self.patient_id:
                #speech = utterance.tiers["PAR"]
                #speech = self.clean_speech_line(speech)
                #full_speech = " ".join([full_speech, speech])
                tokens = utterance.tokens
                for token in tokens:
                    if token.word != "POSTCLITIC":
                        full_speech = " ".join([full_speech, token.word])
        return {"id": id_, "diagnosis": diagnosis, "speech": full_speech.strip()}

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
