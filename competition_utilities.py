from __future__ import division
from collections import Counter
import csv
import dateutil
import numpy as np
import os
import pandas as pd

data_path = "./data/"
submissions_path = "./submission/"
if not data_path or not submissions_path:
    raise Exception("Set the data and submission paths in competition_utilities.py!")

def parse_date_maybe_null(date):
    if date:
        return dateutil.parser.parse(date)
    return None

df_converters = {"PostCreationDate": dateutil.parser.parse,
                 "OwnerCreationDate": dateutil.parser.parse}
                # "PostClosedDate": parse_date_maybe_null}

def get_reader(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return reader

def get_header(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return header

def get_closed_count(file_name):
    return sum(1 for q in iter_closed_questions(file_name))

def get_open_count(file_name):
    return sum(1 for q in iter_open_questions(file_name))

def iter_closed_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=10000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] != "open"].iterrows())

def iter_open_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=10000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] == "open"].iterrows())

def get_dataframe(file_name="train-sample.csv"):
    return pd.io.parsers.read_csv(os.path.join(data_path, file_name), converters = df_converters)

def get_priors(file_name):
    closed_reasons = [r[14] for r in get_reader(file_name)]
    closed_reason_counts = Counter(closed_reasons)
    reasons = sorted(closed_reason_counts.keys())
    total = len(closed_reasons)
    priors = [closed_reason_counts[reason]/total for reason in reasons]
    return priors

def write_sample(file_name, header, sample):
    writer = csv.writer(open(os.path.join(data_path, file_name), "w"), lineterminator="\n")
    writer.writerow(header)
    writer.writerows(sample)

def update_prior(old_prior,  old_posterior, new_prior):
    evidence_ratio = (old_prior*(1-old_posterior)) / (old_posterior*(1-old_prior))
    new_posterior = new_prior / (new_prior + (1-new_prior)*evidence_ratio)
    return new_posterior

def cap_and_update_priors(old_priors, old_posteriors, new_priors, epsilon):
    old_posteriors = cap_predictions(old_posteriors, epsilon)
    old_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), old_priors)
    new_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), new_priors)
    evidence_ratio = (old_priors*(1-old_posteriors)) / (old_posteriors*(1-old_priors))
    new_posteriors = new_priors / (new_priors + (1-new_priors)*evidence_ratio)
    new_posteriors = cap_predictions(new_posteriors, epsilon)
    return new_posteriors

def cap_predictions(probs, epsilon):
    probs[probs>1-epsilon] = 1-epsilon
    probs[probs<epsilon] = epsilon
    row_sums = probs.sum(axis=1)
    probs = probs / row_sums[:, np.newaxis]
    return probs

def write_submission(file_name, predictions):
    writer = csv.writer(open(os.path.join(submissions_path, file_name), "w"), lineterminator="\n")
    writer.writerows(predictions)

def get_actual(labels):
    label_names = {'not a real question': 0, 'not constructive': 1, 'off topic': 2, 'open': 3,'too localized': 4}
    actual = pd.DataFrame(np.zeros((len(labels), len(label_names)), dtype = np.int))
    for index in range(len(labels)):
        actual.iat[index, label_names[labels[index]]] = 1
    return actual.values

def get_log_loss(actual, predictions, epsilon):
    predictions = cap_predictions(predictions, epsilon)
    log_loss = -np.sum(np.sum(np.multiply(actual, np.log(predictions)))) / actual.shape[0]
    return log_loss
