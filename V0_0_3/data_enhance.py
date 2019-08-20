%matplotlib inline
import torch as t
from torchvision import transforms 
import random


def image_transforms(size=224):
    fr_transforms = transforms.Compose([
                        transforms.Resize(size),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(45),
                        transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.3, hue=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                    ])
    return fr_transforms

def enhance_transforms():
    possibility = random.randint(0, 19)
    if possibility % 20 ==0:#0.5%概率不发生变化,相当于数据集扩容20倍
        output_transforms = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        return output_transforms
    else:
        possibility2 = random.randint(0, 3)
        if possibility2%5==0:#20%概率发生不同比例放缩
            return image_transforms(224)
        elif possibility2%5==1:
            return image_transforms(245)#面积放大1.2倍
        elif possibility2%5==2:
            return image_transforms(274)#面积放大1.5倍
        elif possibility2%5==3:
            return image_transforms(316)#面积放大2倍


def transform_standard():
    output_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    return output_transforms
    

