PID_DEPRELS = frozenset(
    [
        "advcl",
        "advmod",
        "amod",
        "appos",
        # "cc",
        "csubj",
        "csubjpass",
        "det",
        "neg",
        "npadvmod",
        "nsubj",
        "nsubjpass",
        "nummod",
        "poss",
        "predet",
        "preconj",
        "prep",
        "quantmod",
        "tmod",
        "vmod",
    ]
)

SID_NSUBJ_NO = frozenset(("it", "this"))
DET_NO = frozenset(("a", "an", "the"))

PATH_TO_LOW_SPECIFICITY = "/Users/kirillkonca/Documents/MAThesis/dementia_prediction/sentences_specificity_filtered.txt"

with open(PATH_TO_LOW_SPECIFICITY, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file]

LOW_SPECIFICITY_SENTENCES = frozenset(sentences)
