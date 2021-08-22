import pandas as pd

labels = pd.read_csv('datasets/train.csv')

while True:
    filename = input('filename: ')
    print(f'name: {filename}')
    print(labels[labels['filename'] == filename])
