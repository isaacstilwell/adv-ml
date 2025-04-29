from model import VGG

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
import random


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Helper function to instantiate new VGG model
def get_vgg_model():
    model = VGG('VGG16')
    return model

# Helper functions to convert between image and tensors

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def img2tensorVGG(pil_img, device):
    return preprocess(pil_img).unsqueeze(0).to(device)

def tensor2imgVGG(tensor_img):
    return to_pil_image(tensor_img.squeeze(0).cpu())

# Helper function for softmax with temperature:

def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=1)

def target_pgd_attack(img, target_class, model, device):
    epsilon = 8/255
    tensor_max = 1
    tensor_min = 0
    lr_initial = 0.01
    max_iter = 200

    source_tensor = img2tensorVGG(img, device)
    modifier = torch.zeros_like(source_tensor, requires_grad=True)

    target_label = torch.tensor([target_class], dtype=torch.long).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(max_iter):
        adv_tensor = torch.clamp(source_tensor + modifier, tensor_min, tensor_max)
        output = model(adv_tensor)
        loss = loss_fn(output, target_label)
        model.zero_grad()
        if modifier.grad is not None:
            modifier.grad.zero_()
        loss.backward()
        grad = modifier.grad
        modifier = modifier - lr_initial * grad.sign()
        modifier = torch.clamp(modifier, min=-epsilon, max=epsilon).detach().requires_grad_(True)

        if i % (max_iter // 10) == 0:
            pred_class = torch.argmax(output, dim=1).item()

            # Optional: uncomment to print loss values:
            # print(f"step: {i} | loss: {loss.item():.4f} | pred class: {classes[pred_class]}")

            if pred_class == target_class:
                break

    adv_tensor = torch.clamp(source_tensor + modifier, tensor_min, tensor_max)
    return tensor2imgVGG(adv_tensor)

