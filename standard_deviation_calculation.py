import os
import pandas as pd


def calculate_std_and_append(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                df = pd.read_csv(file_path)

                std_values = df[['Accuracy', 'Precision', 'Recall', 'F1']].std()

                std_row = pd.DataFrame({
                    'no': ['std_dev'],
                    'Feature': ['-'],
                    'Accuracy': [std_values['Accuracy']],
                    'Precision': [std_values['Precision']],
                    'Recall': [std_values['Recall']],
                    'F1': [std_values['F1']]
                })

                df = pd.concat([df, std_row], ignore_index=True)

                df.to_csv(file_path, index=False)
                print(f'Processed file: {file_path}')


folders = ['./results/folds', './results_deep/folds']

for folder in folders:
    calculate_std_and_append(folder)
