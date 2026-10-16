"""
Step 2: NLP Preprocessing Pipeline

Converts raw financial text into clean tokens ready for ML models.

"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)


# Financial domain vocabulary 
# These words are HIGH-SIGNAL for financial sentiment
FINANCIAL_SIGNAL_WORDS = {
    # Bullish signals
    "bullish", "rally", "surge", "soar", "outperform", "upgrade", "beat",
    "record", "growth", "profit", "gain", "rise", "rose", "risen", "strong",
    "positive", "beat", "exceed", "boost", "high", "upside", "optimistic",

    # Bearish signals
    "bearish", "plunge", "crash", "slump", "miss", "downgrade", "loss",
    "decline", "fall", "fell", "fallen", "weak", "negative", "cut", "reduce",
    "lower", "concern", "risk", "debt", "loss", "default", "bankruptcy",

    # Financial events
    "merger", "acquisition", "ipo", "dividend", "buyback", "recall",
    "lawsuit", "settlement", "fine", "investigation", "restructuring",
}

# Standard English stop words (selectively overriding these for finance)
_STOP_WORDS = set(stopwords.words("english"))

EFFECTIVE_STOP_WORDS = _STOP_WORDS - FINANCIAL_SIGNAL_WORDS - {"no", "not", "nor"}


class FinancialTextPreprocessor:

    # A reusable preprocessing pipeline for financial text.

    def __init__(self, remove_numbers: bool = False):

        # remove_numbers: Whether to strip numbers.
        #                    Default False — numbers like '14%' and '$2.5B'
        #                   can signal magnitude, so I keep them.
        
        self.lemmatizer      = WordNetLemmatizer()
        self.remove_numbers  = remove_numbers
        self.stop_words      = EFFECTIVE_STOP_WORDS

    def clean(self, text: str) -> str:
        # Full preprocessing pipeline. Returns a clean string.

        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove URLs and emails (noise)
        text = re.sub(r"http\S+|www\S+|@\S+|\S+@\S+", " ", text)

        # Step 3: Remove most punctuation but keep $ and %
        # LESSON: re.sub(pattern, replacement, string)
        text = re.sub(r"[^\w\s$%]", " ", text)   # keep word chars, spaces, $, %

        # Step 4: Optionally remove standalone numbers (not financial amounts)
        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)

        # Step 5: Tokenize - split into individual words
        tokens = word_tokenize(text)

        # Step 6: Filter stop words (using the finance-aware set)
        # and very short tokens (single chars are usually noise)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 7: Lemmatize - reduce words to their base form
        # 'earnings' -> 'earning', 'declined' -> 'decline'
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def clean_batch(self, texts) -> list:
        # Process a list of texts.
        return [self.clean(t) for t in texts]


def show_preprocessing_examples(preprocessor: FinancialTextPreprocessor) -> None:

    # Demonstrate the preprocessing on real sentences.
    
    examples = [
        "The company's earnings rose 14% on strong demand for its products.",
        "$ESI on lows, down $1.50 to $2.50 BK a real possibility",
        "Analysts upgraded the stock to BUY citing bullish revenue growth outlook.",
        "The firm announced a restructuring plan, cutting 2,000 jobs.",
        "According to the Finnish-Russian Chamber of Commerce, all major companies are operating.",
    ]

    print("PREPROCESSING EXAMPLES")
    for text in examples:
        cleaned = preprocessor.clean(text)
        print(f"ORIGINAL : {text}")
        print(f"CLEANED  : {cleaned}")
        print(f"TOKENS   : {cleaned.split()}")


if __name__ == "__main__":
    preprocessor = FinancialTextPreprocessor()
    show_preprocessing_examples(preprocessor)

    # Show what happens to key financial signal words
    print("\nFINANCIAL SIGNAL WORD PRESERVATION TEST")
    test = "The stock is bullish after the bearish downgrade and rally"
    print(f"Input  : {test}")
    print(f"Output : {preprocessor.clean(test)}")
    print(" 'bullish', 'bearish', 'downgrade', 'rally' preserved")
