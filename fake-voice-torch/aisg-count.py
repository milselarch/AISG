import re

from trainer import Trainer

trainer = Trainer(cache_threshold=20)
x_test = trainer.x_test

aisg_test = []
for file_path in x_test:
    if type(file_path) is str:
        name = re.search('[a-zA-Z0-9]+\\.', file_path)
        name = name[0][:-1]
        aisg_test.append(name)

print(aisg_test)
print('TEST AISG VIDEOS', len(aisg_test))
aisg_test_str = '\n'.join(aisg_test)
open('aisg-test.txt', 'w').write(aisg_test_str)