from efficientnet_pytorch import EfficientNet
from pytorch_modelsize import SizeEstimator
from torchvision import transforms
from torchsummary import summary
from effnetv2 import effnetv2_m

model = effnetv2_m().to('cuda')
# model = EfficientNet.from_name('efficientnet-b4')
summary(model, input_size=(3, 224, 224))