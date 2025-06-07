import csv
import re


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences

    Args:
        text (str): Text

    Returns:
        list[str]: List of sentences
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s]


def process_csv(input_csv: str, output_txt: str, text_column="speech") -> None:
    """
    Reads dataset and writes every sentence from it to a new line in a file.
    This is needed in order to perform specificity scoring

    Args:
        input_csv (str): Path to csv-file with dataset
        output_txt (str): Path to output txt-file
        text_column (ste): Column with text

    Returns:
        None: Function writes sentences in a file. Does not return anything
    """
    with open(input_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        with open(output_txt, "w", encoding="utf-8") as txt_file:
            for row in reader:
                if text_column in row:
                    sentences = split_into_sentences(row[text_column])
                    for sentence in sentences:
                        txt_file.write(sentence + "\n")


def filter_sentences_by_score(
    sentences_file: str, scores_file: str, output_txt: str, threshold: float = 0.01
) -> None:
    """
    Filters out sentences with specificity below the threshold. According to Sirts et al. (2017)

    Args:
        sentences_file (str): Path to file with sentences to filter
        scores_file (str): Path to file with specificity scores
        output_txt (str): Path to file with filtered out sentences
        threshold (float): Threshold to filter out sentences by specificity

    Returns:
        None: function writes filtered out sentences in a file. Nothing is returned
    """
    with open(sentences_file, "r", encoding="utf-8") as sent_file, open(
        scores_file, "r", encoding="utf-8"
    ) as score_file, open(output_txt, "w", encoding="utf-8") as txt_file:
        sentences = [line.strip() for line in sent_file.readlines()]
        scores = [float(line.strip()) for line in score_file.readlines()]

        for sentence, score in zip(sentences, scores):
            if score < threshold:
                txt_file.write(sentence + "\n")
