from sklearn.metrics import roc_auc_score
from pprint import pprint
from navec import Navec
from razdel import tokenize, sentenize
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')


def multiclass_roc_auc(y_true, y_pred):
    if len(set(y_true)) == 2:
        r = roc_auc_score(y_true, y_pred[:, 1])
    else:
        r = roc_auc_score(y_true, y_pred[:, 1], multi_class='ovr')
    return r


def print_most_incorrect(df, n=10):
    most_incorrect = df.loc[df.apply(lambda row: row['predictions'][row['target']], axis=1).sort_values()[:n].index]
    for _, incorrect_row in most_incorrect.iterrows():
        pprint(incorrect_row['text'])
        print(incorrect_row[['target', 'coloring', 'predictions']])
        print()


def vectorize_text(text):
    r = 0
    for token in tokenize(text):
        word = token.text.lower()
        if word in navec:
            r = r + navec[word]
    return r