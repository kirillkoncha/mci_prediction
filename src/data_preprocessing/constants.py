IGNORE_TOKENS = frozenset(("POSTCLITIC", "‡"))

IGNORE_POS = frozenset(("n:let", "beg", "end", "on", "co"))

NEOLOGISMS = frozenset(("uni", "neo"))

PUNCT_REPLACEMENTS = {
    '+"/.': ".",
    "+/.": ".",
    "+//?": "?",
    "+/?": "?",
    '+\\"/.': ".",
    "+//.": ".",
    '+\\".': ".",
}
