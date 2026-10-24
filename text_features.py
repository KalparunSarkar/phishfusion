"""Character-level text representations for URL strings and HTML content.

Implements the char-level TF-IDF representation used by the base paper
(Aljofey et al., 2022) in two flavours:

* ``CharTfidf`` - sklearn ``TfidfVectorizer`` with character n-grams. This is the
  representation the base paper found effective on noisy HTML fragments and
  plaintext.
* ``HashingCharEmbedding`` - a dependency-free, vocabulary-free alternative using
  feature hashing, useful for streaming / very large corpora.

Both expose ``fit``/``transform`` and ``fit_transform`` so they can be swapped
transparently in the pipeline.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from config import CHAR_MAX_FEATURES_TEXT, CHAR_MAX_FEATURES_URL, CHAR_NGRAM_RANGE


class CharTfidf:
    def __init__(self, ngram_range=CHAR_NGRAM_RANGE, max_features=CHAR_MAX_FEATURES_URL,
                 min_df=2, sublinear_tf=True):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
        )

    def fit(self, texts):
        self.vectorizer.fit([t or "" for t in texts])
        return self

    def transform(self, texts):
        return self.vectorizer.transform([t or "" for t in texts])

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform([t or "" for t in texts])

    def n_features(self):
        try:
            return len(self.vectorizer.get_feature_names_out())
        except Exception:
            return 0

    def feature_names(self):
        try:
            return list(self.vectorizer.get_feature_names_out())
        except Exception:
            return None


class HashingCharEmbedding:
    def __init__(self, n_features=CHAR_MAX_FEATURES_URL, ngram_range=CHAR_NGRAM_RANGE):
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def fit(self, texts):
        return self

    def transform(self, texts):
        return self.vectorizer.transform([t or "" for t in texts])

    def fit_transform(self, texts):
        return self.transform(texts)

    def n_features(self):
        return self.vectorizer.n_features

    def feature_names(self):
        return None


def _make(kind, max_features):
    if kind == "hash":
        return HashingCharEmbedding(n_features=max_features)
    return CharTfidf(max_features=max_features)


class TextFeatureBuilder:
    """Maintains separate char representations for URL strings and page text."""

    def __init__(self, kind="tfidf", url_max=CHAR_MAX_FEATURES_URL,
                 text_max=CHAR_MAX_FEATURES_TEXT):
        self.kind = kind
        self.url_vec = _make(kind, url_max)
        self.text_vec = _make(kind, text_max)
        self._dense = True

    def fit(self, urls, page_texts):
        self.url_vec.fit(urls)
        self.text_vec.fit(page_texts)
        return self

    def transform(self, urls, page_texts):
        xu = self.url_vec.transform(urls)
        xt = self.text_vec.transform(page_texts)
        return sparse.hstack([xu, xt]).tocsr()

    def fit_transform(self, urls, page_texts):
        self.fit(urls, page_texts)
        return self.transform(urls, page_texts)

    def n_features(self):
        return self.url_vec.n_features() + self.text_vec.n_features()


def to_dense(matrix):
    return np.asarray(matrix.todense()) if sparse.issparse(matrix) else np.asarray(matrix)


if __name__ == "__main__":
    urls = [
        "http://secure-login.paypal.com.verify.xyz/webscr?cmd=login",
        "https://www.google.com/search?q=hello",
        "http://bit.ly/2xYz",
    ]
    texts = ["Dear user verify your account immediately", "Search results page", "short link"]
    tb = TextFeatureBuilder()
    X = tb.fit_transform(urls, texts)
    print("char feature matrix:", X.shape, "| total features:", tb.n_features())
