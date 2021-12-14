import json
import torch

from PIL import Image
from torchvision import transforms
from effnetv2 import *
from efficientnet_pytorch import EfficientNet

"""
medium: 400.85546875
large: 705.61083984375
extra-large: 1053.1572265625
"""

model = effnetv2_m(num_classes=1).to('cuda')

# Preprocess image
tfms = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )]
)

image_path = 'test_images/0-11.jpg'
imgs = torch.cat([
    tfms(Image.open(image_path)).unsqueeze(0)
    for k in range(32)
])

imgs = imgs.to('cuda')
print(imgs.shape)  # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
# labels_map = json.load(open('labels_map.txt'))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(imgs)

print('mem usage')
allocated = torch.cuda.max_memory_allocated()
print(allocated / (1024 ** 2))
# allocated = aft_allocated - pre_allocated
# print((aft_allocated - pre_allocated) / (1024 ** 2))

# Print predictions
"""
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print(f'predict [{idx}] = {prob}')
"""

print('OUTPUT', outputs)