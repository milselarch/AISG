import torch

from PIL import Image
from torchvision import transforms
from MesoTrainer import MesoTrainer

model_path = 'saves/models/211022-0001.pt'
# face predictions exported to ../stats/face-predictions-211106-0928.csv

trainer = MesoTrainer()
trainer.load_model(model_path)
model = trainer.model

tfms = transforms.Compose([
    transforms.Resize(256), transforms.ToTensor(),
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

model.eval()
with torch.no_grad():
    outputs = model(imgs)

print('mem usage')
allocated = torch.cuda.max_memory_allocated()
print(allocated / (1024 ** 2))
