PATH_TO_LOW_SPECIFICITY = "./data/sentences_specificity_filtered.txt"
PATH_TO_WORD_CLUSTERS = "./data/cluster_words.csv"

with open(PATH_TO_LOW_SPECIFICITY, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file]

LOW_SPECIFICITY_SENTENCES = frozenset(sentences)
