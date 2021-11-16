import pandas as pd

labels = pd.read_csv('datasets/train.csv')

# fd1404019e28214f.mp4
# fd729db3f9584c66.mp4
print('label finder')

while True:
    filename = input('filename: ')
    print(f'name: {filename}')
    print(labels[labels['filename'] == filename])
