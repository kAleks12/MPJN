import argparse
import csv
import logging
import os

import numpy as np
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoConfig, \
    AutoModelForSequenceClassification, DataCollatorWithPadding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load datasets
def load_datasets():
    return [
        (load_dataset("thehamkercat/telegram-spam-ham", split="train"), "thehamkercat/telegram-spam-ham", 'text_type',
         'text'),
        (load_dataset("FredZhang7/all-scam-spam", split="train"), "FredZhang7/all-scam-spam", 'is_spam', 'text'),
        (load_dataset("prithivMLmods/Spam-Text-Detect-Analysis", split="train"),
         "prithivMLmods/Spam-Text-Detect-Analysis", 'Category', 'Message')
    ]


def tokenize(tokenizer, texts, labels):
    def data_generator(texts, labels):
        for text, label in zip(texts, labels):
            yield {"text": text, "labels": label}

    dataset = Dataset.from_generator(
        lambda: data_generator(texts, labels)
    )

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=64)
        tokenized["labels"] = examples["labels"]
        return tokenized

    return dataset.map(tokenize_function, batched=True)


def train_and_evaluate(model_name, texts, labels, out_dir, dataset_name, file_suffix, feature_name, extra_logs=False,
                       params=None):
    model_name_safe = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logging.info("Tokenizing texts...")
    tokenized_dataset = tokenize(tokenizer, texts, labels)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, dataset_name),
        learning_rate=params.get('learning_rate', 2e-5),
        per_device_train_batch_size=params.get('batch_size', 64),
        per_device_eval_batch_size=params.get('batch_size', 64),
        num_train_epochs=params.get('epochs', 1),
        weight_decay=params.get('weight_decay', 0.01),
        save_strategy='no',
        logging_strategy='epoch',
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        predictions = pred.predictions.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=36851234)
    logging.info("Starting cross-validation...")
    for i, (train_idx, val_idx) in enumerate(rskf.split(tokenized_dataset['text'], tokenized_dataset['labels'])):
        train_dataset = tokenized_dataset.select(train_idx)
        val_dataset = tokenized_dataset.select(val_idx)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        partial_results = trainer.evaluate()
        accuracies.append(partial_results['eval_accuracy'])
        precisions.append(partial_results['eval_precision'])
        recalls.append(partial_results['eval_recall'])
        f1_scores.append(partial_results['eval_f1'])

        logging.info(f'Results for fold no. {i}: {partial_results}')
        if extra_logs:
            with open(os.path.join(out_dir, 'folds', f'{dataset_name}_{file_suffix}_{feature_name}_{model_name_safe}.csv'), 'a',
                      newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # Write header if file is empty
                    writer.writerow(['no', 'Accuracy', 'Precision', 'Recall', 'F1'])
                writer.writerow(
                    [i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)])

    logging.info(f"Metrics (RSKF summary) -> "
                 f"avg accuracy: {np.mean(accuracies)}; avg precision: {np.mean(precisions)}; "
                 f"avg recall: {np.mean(recalls)}; avg f1: {np.mean(f1_scores)}")

    with open(os.path.join(out_dir, f'results_{file_suffix}.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header if file is empty
            writer.writerow(['Dataset', 'Model', 'Feature', 'Accuracy', 'Precision', 'Recall', 'F1'])
        writer.writerow([dataset_name, model_name, feature_name, np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                         np.mean(f1_scores)])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep learning Models Evaluation Script")
    parser.add_argument("--eval", type=str, choices=["true", "false"],
                        default="true", help="Evaluate deep learning models. Default is true.")
    parser.add_argument("--params", type=str, choices=["true", "false"],
                        default="true", help="Evaluate hyperparameters. Default is true.")
    parser.add_argument("--extra_logs", type=str, choices=["true", "false"],
                        default="false", help="Save logs for each fold. Default is true.")
    args = parser.parse_args()
    extra_logs = True if args.extra_logs == "true" else False

    output_dir = "results_deep"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'folds'), exist_ok=True)

    if torch.cuda.is_available():
        logging.info(f"CUDA is available. PyTorch is using {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA is not available. PyTorch is using the CPU.")

    datasets = load_datasets()
    models = ["mshenoda/roberta-spam", "prithivMLmods/Spam-Bert-Uncased"]
    learning_rates = [1e-5, 3e-5]
    batch_sizes = [32, 128]
    num_epochs = [0.5, 1.5]
    weight_decays = [0.1, 0.001]

    for dataset, name, label_column, text_column in datasets:
        name = name.replace("/", "_")
        logging.info(f"Evaluating {name} dataset...")
        texts, labels = dataset[text_column], dataset[label_column]
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        if args.eval == "true":
            for model in models:
                logging.info(f"Evaluating {model} model...")
                train_and_evaluate(model, texts, labels, output_dir, name, "eval", "evaluation", extra_logs, {})

        if args.params == 'true':
            for model in models:
                for learning_rate in learning_rates:
                    logging.info(f"Evaluating with learning_rate: {learning_rate}")
                    train_and_evaluate(model, texts, labels, output_dir, name, "param",
                                       f'learning_rate_{learning_rate}',
                                       extra_logs, {'learning_rate': learning_rate})

                # for batch_size in batch_sizes:
                #     train_and_evaluate(model, texts, labels, output_dir, name, "param",
                #                        f'batch_size_{batch_size}', extra_logs, {'batch_size': batch_size})
                #
                # for epoch in num_epochs:
                #     train_and_evaluate(model, texts, labels, output_dir, name, "param", f'epochs_{epoch}', extra_logs,
                #                        {'num_epochs': epoch})

                for weight_decay in weight_decays:
                    train_and_evaluate(model, texts, labels, output_dir, name, "param", f'weight_decay_{weight_decay}',
                                       extra_logs, {'weight_decay': weight_decay})
