import argparse
import logging
import csv
import os

import numpy as np
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Custom transformer for Word2Vec embeddings
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, documents, y=None):
        tokenized_docs = [doc.split() for doc in documents]
        self.model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4, seed=42)
        return self

    def transform(self, documents):
        tokenized_docs = [doc.split() for doc in documents]
        return np.array(
            [np.mean([self.model.wv[word] for word in doc if word in self.model.wv] or [np.zeros(100)], axis=0) for doc
             in tokenized_docs])


# Load datasets from Hugging Face
def load_datasets():
    return [
        (load_dataset("thehamkercat/telegram-spam-ham", split="train"), "thehamkercat/telegram-spam-ham", 'text_type',
         'text'),
        (load_dataset("FredZhang7/all-scam-spam", split="train"), "FredZhang7/all-scam-spam", 'is_spam', 'text'),
        (load_dataset("prithivMLmods/Spam-Text-Detect-Analysis", split="train"),
         "prithivMLmods/Spam-Text-Detect-Analysis", 'Category', 'Message')
    ]


def preprocess_dataset(dataset, text_column, label_column):
    texts = dataset[text_column]
    labels = dataset[label_column]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return texts, labels


def evaluate_model_cross_val(model, X, y, dataset_name, file_suffix, feature_name, out_dir="", extra_logs=False):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=36851234)

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions))
        recalls.append(recall_score(y_test, predictions))
        f1_scores.append(f1_score(y_test, predictions))
        logging.info(f"Cross-Validation no. {i} Complete. "
                     f"Results: accuracy: {accuracies[-1]}; precision: {precisions[-1]}; "
                     f"recall: {recalls[-1]}; f1: {f1_scores[-1]}")
        if extra_logs:
            with open(os.path.join(out_dir, 'folds', f'{dataset_name}_{file_suffix}.csv'), 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # Write header if file is empty
                    writer.writerow(['no', 'Feature', 'Accuracy', 'Precision', 'Recall', 'F1'])
                writer.writerow([i, feature_name, np.mean(accuracies), np.mean(precisions), np.mean(recalls),np.mean(f1_scores)])

    logging.info(f"Metrics (RSKF summary) -> "
                 f"avg accuracy: {np.mean(accuracies)}; avg precision: {np.mean(precisions)}; "
                 f"avg recall: {np.mean(recalls)}; avg f1: {np.mean(f1_scores)}")

    with open(os.path.join(out_dir, f'results_{file_suffix}.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header if file is empty
            writer.writerow(['Dataset', 'Feature', 'Accuracy', 'Precision', 'Recall', 'F1'])
        writer.writerow([dataset_name, feature_name, np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                         np.mean(f1_scores)])



def compare_feature_extraction_methods(ensemble_model, texts, labels, dataset_name, out_dir="", extra_logs=False):
    feature_extractors = {
        "TF-IDF": TfidfVectorizer(max_features=100),
        "Bag of Words": CountVectorizer(max_features=100),
        "Word Embedding": Word2VecTransformer()
    }

    for name, extractor in feature_extractors.items():
        logging.info(f"Using {name} extractor...")
        if isinstance(extractor, TfidfVectorizer) or isinstance(extractor, CountVectorizer):
            X = extractor.fit_transform(texts).toarray()
        else:
            X = extractor.fit_transform(texts)
        evaluate_model_cross_val(
            ensemble_model, X, labels, dataset_name, "extraction", name, out_dir, extra_logs
        )


def test_voting_methods(texts, labels, dataset_name, out_dir="", extra_logs=False):
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()

    # Majority Voting
    logging.info("Using Majority Voting...")
    majority_voting_model = VotingClassifier(
        estimators=[("RandomForest", RandomForestClassifier(random_state=42)),
                    ("SVC", SVC(probability=True, random_state=42)),
                    ("LogisticRegression", LogisticRegression(random_state=42)),
                    ("NaiveBayes", GaussianNB())],
        voting="hard"
    )
    evaluate_model_cross_val(
        majority_voting_model, X, labels, dataset_name, "voting", "majority", out_dir, extra_logs
    )

    # Weighted Voting
    logging.info("Using Weighted Voting...")
    weighted_voting_model = VotingClassifier(
        estimators=[("RandomForest", RandomForestClassifier(random_state=42)),
                    ("SVC", SVC(probability=True, random_state=42)),
                    ("LogisticRegression", LogisticRegression(random_state=42)),
                    ("NaiveBayes", GaussianNB())],
        voting="soft",
        weights=[2, 1, 1, 1]  # Assigning higher weight to RandomForest
    )
    evaluate_model_cross_val(
        weighted_voting_model, X, labels, dataset_name, "voting", "wight", out_dir, extra_logs
    )

    # Meta Model
    logging.info("Using Meta Model...")
    base_estimators = [
        ("RandomForest", RandomForestClassifier(random_state=42)),
        ("SVC", SVC(probability=True, random_state=42)),
        ("LogisticRegression", LogisticRegression(random_state=42)),
        ("NaiveBayes", GaussianNB())
    ]
    meta_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(),
        passthrough=False
    )
    evaluate_model_cross_val(
        meta_model, X, labels, dataset_name, "voting", "meta", out_dir, extra_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Model Evaluation Script")
    parser.add_argument("--eval", type=str, choices=["true", "false"],
                        default="true", help="Evaluate ensemble. Default is true.")
    parser.add_argument("--features", type=str, choices=["true", "false"],
                        default="true", help="Evaluate feature extraction methods. Default is false.")
    parser.add_argument("--voting", type=str, choices=["true", "true"],
                        default="true", help="Evaluate voting methods. Default is false.")
    parser.add_argument("--extra_logs", type=str, choices=["true", "false"],
                        default="false", help="Save logs for each fold. Default is true.")
    args = parser.parse_args()

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "folds"), exist_ok=True)

    extra_logs = True if args.extra_logs == "true" else False

    datasets = load_datasets()

    ensemble_model = VotingClassifier(
        estimators=[("RandomForest", RandomForestClassifier(random_state=42)),
                    ("SVC", SVC(probability=True, random_state=42)),
                    ("LogisticRegression", LogisticRegression(random_state=42, max_iter=500)),
                    ("NaiveBayes", GaussianNB())],
        voting="soft"
    )

    for dataset, name, label_column, text_column in datasets:
        name = name.replace("/", "_")

        logging.info(f"Evaluating dataset '{name}...'")
        texts, labels = preprocess_dataset(dataset, text_column, label_column)

        if args.eval == "true":
            logging.info("Evaluating ensemble model...")
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(texts).toarray()
            evaluate_model_cross_val(
                ensemble_model, X, labels, name, "eval", "evaluation", output_dir, extra_logs=extra_logs
            )

        if args.features == "true":
            logging.info("Running feature extraction comparison...")
            compare_feature_extraction_methods(ensemble_model, texts, labels, name, output_dir, extra_logs=extra_logs)

        if args.voting == "true":
            logging.info("Running voting method comparison...")
            test_voting_methods(texts, labels, name, output_dir, extra_logs=extra_logs)
