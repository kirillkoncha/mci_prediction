PID_DEPRELS = frozenset(
    (
        "advcl",
        "advmod",
        "amod",
        "appos",
        # "cc", is exlcuded in the paper
        "csubj",
        "csubjpass",
        "det",
        "neg",
        "npadvmod",
        "nsubj",
        "nummod",
        "poss",
        "predet",
        "preconj",
        "prep",
        "quantmod",
        "tmod",
        "vmod",
    )
)

SID_NSUBJ_NO = frozenset(("it", "this"))
DET_NO = frozenset(("a", "an", "the"))

LOW_SPECIFICITY_SENTENCES = frozenset(())
