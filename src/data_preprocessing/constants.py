IGNORE_TOKENS = frozenset(("POSTCLITIC", "‡"))
#        "uh",
#        "um",
#        "er",
#        "ah",
#        "mhm",
#        "oy",
#        "huh",
#        "oh",
#        "uhoh",
#        "yeah",
#        "whew",
#        "shh",
#        "hm",
#        "uhhuh",
#        "oh_boy",
#        "oh_my_gosh",
#        "hmhunh",
#        "oh_no",
#        "ho",
#        "wahoo",
#        "shh",
#        "oh_gosh",
#        "gee",
#        "wowie",
#        "whew",
#        "oop",
#        "mm",
#        "wow"
#    )
# )

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
