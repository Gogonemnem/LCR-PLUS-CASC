"""End-to-end weakly-supervised labeling: vocab -> extract -> score -> label."""
from .extracter import Extracter
from .labeler import Labeler
from .mlm import MLMScorer
from .score_computer import ScoreComputer
from .vocab import VocabGenerator


def run_labeling():
    """Execute the full labeling pipeline for the configured domain."""
    scorer = MLMScorer()

    generator = VocabGenerator(scorer=scorer, save_results=True)
    try:
        aspect_vocabularies, sentiment_vocabularies = generator.from_folder()
    except FileNotFoundError:
        aspect_vocabularies, sentiment_vocabularies = generator()

    sentences, aspects, opinions = Extracter()()
    ScoreComputer(aspect_vocabularies, sentiment_vocabularies, scorer=scorer)(
        sentences, aspects, opinions
    )
    Labeler()()
