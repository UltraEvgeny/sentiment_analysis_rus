import natasha
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from funcs import *
import numpy as np


df = pd.read_csv('test_corpus1.csv')
df = df[df['text'].notnull()]#.sample(100)

sentiment_mapping = {'bad': 0, 'neutral': 0, 'good': 1}
df['target'] = df['coloring'].map(sentiment_mapping)

print(df['target'].value_counts())


df['vector'] = df['text'].map(vectorize_text)
df_train, df_test = train_test_split(df[['text', 'target', 'coloring', 'vector']].copy(), test_size=0.3)
x_train, x_test = np.array(df_train['vector'].to_list()), np.array(df_test['vector'].to_list())
y_train, y_test = df_train['target'], df_test['target']
clf = LogisticRegression(multi_class='ovr', class_weight=(1/df_train['target'].value_counts()).to_dict(), max_iter=10000)
clf.fit(x_train, y_train)

y_pred_train, y_pred_test = clf.predict_proba(x_train), clf.predict_proba(x_test)
df_train.insert(0, 'predictions', [x.round(3).tolist() for x in y_pred_train])
df_test.insert(0, 'predictions', [x.round(3).tolist() for x in y_pred_test])
df_test['predictions'] = [x.round(3).tolist() for x in y_pred_test]
y_pred_train_label, y_pred_test_label = y_pred_train.argmax(axis=1), y_pred_test.argmax(axis=1)
print(multiclass_roc_auc(y_train, y_pred_train))
print(multiclass_roc_auc(y_test, y_pred_test))

confusion = pd.DataFrame({'true': y_test, 'pred': y_pred_test_label}).groupby(['true', 'pred']).size() \
    .reset_index(name='fraction').pivot(index='true', columns='pred', values='fraction') \
    .reindex(index=set(sentiment_mapping.values()), columns=set(sentiment_mapping.values())).fillna(0)
confusion /= confusion.sum().sum()
print(confusion)
print(y_pred_train)
print(y_pred_test)

print_most_incorrect(df_train)
print_most_incorrect(df_test)

a = 3
