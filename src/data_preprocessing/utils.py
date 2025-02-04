import csv
import re


def split_into_sentences(text: str):
    # Regex to split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s]


def process_csv(input_csv: str, output_txt: str, text_column="speech"):
    with open(input_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        with open(output_txt, "w", encoding="utf-8") as txt_file:
            for row in reader:
                if text_column in row:
                    sentences = split_into_sentences(row[text_column])
                    for sentence in sentences:
                        txt_file.write(sentence + "\n")


def filter_sentences_by_score(sentences_file, scores_file, output_txt):
    with open(sentences_file, "r", encoding="utf-8") as sent_file, open(
        scores_file, "r", encoding="utf-8"
    ) as score_file, open(output_txt, "w", encoding="utf-8") as txt_file:
        sentences = [line.strip() for line in sent_file.readlines()]
        scores = [float(line.strip()) for line in score_file.readlines()]

        for sentence, score in zip(sentences, scores):
            if score < 0.01:
                txt_file.write(sentence + "\n")
