import re
from itertools import groupby
import nltk
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
import competition_utilities as cu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

train_file = 'train-sample.csv'
test_file = 'test_data.csv'
output_test_path = 'data/test_p.csv'
output_train_path = 'data/train_p.csv'

RE_DIGIT = re.compile(r'\d+')
RE_URL = re.compile(r'https?://')
RE_NONWORD  = re.compile(r'[A-Z\d]+')

def alpha_num_filter(string):
    return re.sub(r'[^\w\s]+', '', string).lower()

def word_filter(string):
    return re.sub(r'\W+', '', string).lower()

def ratio(x, y):
    if y != 0:
        return x / float(y)
    else:
        return 0

def get_predictions(tf_idf, train_fea, train_label, test_fea):
    train_fea_m = tf_idf.transform(train_fea)
    test_fea_m = tf_idf.transform(test_fea)
    rf = RandomForestClassifier(n_estimators=50, verbose=2, n_jobs=-1)
    rf.fit(train_fea_m.toarray(), train_label)
    return rf.predict_proba(test_fea_m.toarray())

def cross_validate_get_predictions(tf_idf, var, train_data):
    train_data["r"] = np.random.uniform(0,1,size=len(train_data))
    for i in range(0,10):
        train_index = ((train_data['r'] < i*0.1) | (train_data['r'] >= (i+1)*0.1))
        test_index = ((train_data['r'] >= i*0.1) & (train_data['r'] < (i+1)*0.1))
        probs = get_predictions(tf_idf, train_data[var][train_index], train_data['label'][train_index], train_data[var][test_index])
        for i in range(1,6):
            train_data[var+'_pred%d'%i][test_index] = probs[:,i-1]

def row_to_features(row):
    title = row['Title']
    body = row['BodyMarkdown']

    lines = body.splitlines()
    code_block = []
    text_block = []
    sentences = []
    features = {}

    for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
        (code_block if is_code else text_block).append('\n'.join(group))

    features['post_creation_date'] = row['PostCreationDate']
    features['owner_creation_date'] = row['OwnerCreationDate']

    features['user_age'] = (features['post_creation_date'] - features['owner_creation_date']).total_seconds()
    features['user_reputation'] = int(row['ReputationAtPostCreation'])
    features['user_good_posts_num'] = int(row['OwnerUndeletedAnswerCountAtPostTime'])
    features['tags'] = [word_filter(row["Tag%d"%i]) for i in range(1,6) if row["Tag%d"%i]]

    features['body_length'] = len(body)
    features['tags_num'] = len(features['tags'])
    features['sentences_num'] = 0
    features['period_num'] = 0
    features['question_mark_num'] = 0
    features['exclam_num'] = 0
    features['istart_num'] = 0
    features['init_cap_num'] = 0
    features['digit_num'] = 0
    features['urls_num'] = 0
    features['nonword_num'] = 0
    features['code_block_num'] = 0
    features['text_block_num'] = 0

    body_words = set()
    for text in text_block:
        for sentence in nltk.sent_tokenize(text):
            features['sentences_num'] += 1
            ss = sentence.strip()
            if ss:
                if ss.endswith('.'):
                    features['period_num'] += 1
                if ss.endswith('?'):
                    features['question_mark_num'] += 1
                if ss.endswith('!'):
                    features['exclam_num'] += 1
                if ss.startswith('I'):
                    features['istart_num'] += 1
                if ss[0].isupper():
                    features['init_cap_num'] += 1
            words = nltk.word_tokenize(alpha_num_filter(sentence))
            body_words |= set(words)
            sentences.append(sentence)
        features['urls_num'] += len(RE_URL.findall(text))
        features['nonword_num'] += len(RE_NONWORD.findall(text))
        features['digit_num'] += len(RE_DIGIT.findall(text))
    features['code_block_num'] = len(code_block)
    features['text_block_num'] = len(text_block)
    features['lines_num'] = len(lines)
    features['text_len'] = sum(len(t) for t in text_block)
    features['code_len'] = sum(len(c) for c in code_block)
    features['mean_code_len'] = np.mean([len(c) for c in code_block]) if code_block else 0
    features['mean_text_len'] = np.mean([len(t) for t in text_block]) if text_block else 0
    features['mean_sentence_len'] = np.mean([len(s) for s in sentences]) if sentences else 0
    features['title_length'] = len(title)

    features['text_code_ratio'] = ratio(features['text_block_num'], features['code_block_num'])
    features['question_ratio'] = ratio(features['question_mark_num'], features['sentences_num'])
    features['exclam_ratio'] = ratio(features['exclam_num'], features['sentences_num'])
    features['period_ratio'] = ratio(features['period_num'], features['sentences_num'])

    # features['body_words'] = list(body_words)
    # features['title_words'] = list(set(nltk.word_tokenize(alpha_num_filter(title))))
    features['body'] = body
    features['title'] = title
    features['label'] = row['OpenStatus']
    features['tags'] = ' '.join(features)

    return features

def get_preprocess_data(data):
    data_p = []
    for i in range(0, len(data)):
        data_p.append(row_to_features(data.ix[i]))
    data_p = pd.DataFrame(data_p)
    return data_p

if __name__=="__main__":
    print("get train data")
    train_data = cu.get_dataframe(train_file)
    print("get test data")
    test_data = cu.get_dataframe(test_file)
    train_data = train_data.replace(np.nan,'', regex=True)
    test_data = test_data.replace(np.nan,'', regex=True)
    train_data_p = get_preprocess_data(train_data)
    test_data_p = get_preprocess_data(test_data)
    text_vars = ['title', 'tags', 'body']
    print("print tf idf")
    for var in text_vars:
        tf_idf = TfidfVectorizer(min_df=2, use_idf=1, smooth_idf=1, sublinear_tf=1, ngram_range=(1,2), norm='l2')
        tf_idf.fit(train_data_p[var].append(test_data_p[var]))
        probs = get_predictions(tf_idf, train_data_p[var], test_data_p[var], train_data_p['label'])

        for i in range(1,6):
            test_data_p[var+'_pred%d'%i] = probs[:, i-1]
            train_data_p[var+'_pred%d'%i] = 0.0
        cross_validate_get_predictions(tf_idf, var, train_data_p)
    header = ['post_creation_date', 'owner_creation_date', 'user_age', 'tags_num', 'sentences_num', 'sentences_num', 'period_num', 'question_mark_num', 'exclam_num', 'istart_num', 'init_cap_num',
    'digit_num', 'urls_num', 'nonword_num', 'code_block_num', 'text_block_num', 'lines_num', 'text_len', 'code_len', 'mean_code_len', 'mean_text_len', 'mean_sentence_len', 'title_length', 'text_code_ratio', 'question_ratio',
    'exclam_ratio', 'period_ratio']
    for var in text_vars:
        for i in range(1,6):
            header.append(var+'_pred%d'%i)
    header.append('label')
    train_data_p[header].to_csv(output_train_path, index = False)
    test_data_p[header].to_csv(output_test_path, index = False)
