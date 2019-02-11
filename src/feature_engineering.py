from data_loader import load_data
from sklearn.pipeline import Pipeline
from nlp_processing import LemmaCountVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# train, test = load_data()
# full_text = list(train.iloc[:, 1].values) + list(test.iloc[:, 1].values)

a = ["this is is a test", "I like This test, what about you? test!! testing", "Estó funciona súper!"]

parameters = {
    'vec__stop_words': (None, 'english'),
    'vec__ngram_range': ((1, 1), (1, 2), (2, 2)),
    'vec__binary': (True, False),
    'vec__min_df': (1, .3, .4, .5),
    'tfidf__norm': ('l1' 'l2'),
    'tfidf__smooth_idf': (True, False)
}



c = LemmaCountVectorizer(strip_accents='unicode', stem=False)
print(c)
xx =c.fit_transform(a)
print(xx.toarray())
print(c.get_feature_names())
