from spacy_conll import init_parser


class UDAnnotator:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.nlp = init_parser(
            lang,
            "udpipe",
            parser_opts={"use_gpu": use_gpu, "verbose": False},
            include_headers=True,
        )

    def annotate_text(self, text: str) -> str:
        """
        Annotates Universal Dependencies in a given text

        Args:
            text (str): Text to be annotated

        Returns:
            conllu.models.SentenceList: Annotated text
        """
        doc = self.nlp(text)
        conll = doc._.conll_str
        return conll
