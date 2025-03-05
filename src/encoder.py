
import torch
from src import resnet

 
ResNet = resnet.ResNet
Bottleneck = resnet.Bottleneck

def CSA(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
   
    return model
 

 
 