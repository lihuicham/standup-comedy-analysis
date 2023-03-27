from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer

class FreqNaiveBayes(GaussianNB):
    def __init__(self, freq_margin = 0, **kwargs):
        """
        Wrapper around GaussianNB that processed the training and test features by removing
        words that appear too often or too infrequent. The new transcript is then vectorised and
        passed to GaussianNB for training
        """
        
        super().__init__(**kwargs)
        self.freq_margin = freq_margin

    def fit(self, X, y, show_shape = False,**kwargs):
        """
        Takes in raw scripts as X and script labels as y.
        Stores the vectoriser used to vectorise the scripts
        Calls the fit method of GaussianNB to perform the fitting
        """
        
        lower, upper = self.freq_margin, 1 - self.freq_margin

        vectorizer = TfidfVectorizer(min_df = lower, max_df = upper)
        train_vec = vectorizer.fit_transform(X).toarray()

        super().fit(train_vec, y)
        self.vectorizer = vectorizer

        if show_shape:
            print(f"Train shape: {train_vec.shape}")

    def predict(self, X, **kwargs):
        """
        Wrapper for predict method of GaussianNB
        Predicts by first processing X in the same way as the training data, then uses
        predict method of GaussianNB to perform predictions
        """

        vectorizer = self.vectorizer
        test_vec = vectorizer.transform(X).toarray()

        return super().predict(test_vec, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Similar to predict, wraps around the predict_proba method of GaussianNB
        """

        vectorizer = self.vectorizer
        test_vec = vectorizer.transform(X).toarray()

        return super().predict_proba(test_vec, **kwargs)
        