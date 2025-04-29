"""
Starter file for HW2, CMSC 25800 Spring 2025
"""

from utils import ResNet18, vgg19
from utils import img2tensorResNet, tensor2imgResNet # for part 1
from utils import img2tensorVGG, tensor2imgVGG # for parts 2 and 3

from PIL import Image
import requests
import io

import numpy as np
import torch

# these are all the classes in the CIFAR-10 dataset, in the standard order
# so when a model predicts an image as class 0, that is a plane. class 1 is a car, class 2 is a bird, etc.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def part_1(
    img: Image,
    target_class: int,
    model: ResNet18,
    device: str | torch.device
) -> Image:
    epsilon = 15.95/255
    iters = 250
    learning_rate = 0.001

    original_img = img2tensorResNet(img, device)
    img_tensor = original_img.clone().detach().requires_grad_(True)
    target_tensor = torch.tensor([target_class], device=device)

    for i in range(iters):
        img_tensor.grad = None
        outputs = model(img_tensor)

        loss = torch.nn.functional.cross_entropy(outputs, target_tensor)

        loss.backward()

        with torch.no_grad():
            img_tensor = img_tensor - learning_rate * img_tensor.grad.sign()
            diff = img_tensor - original_img
            diff = torch.clamp(diff, -epsilon, epsilon)
            img_tensor = torch.clamp(original_img + diff, -1, 1).requires_grad_()

    img = tensor2imgResNet(img_tensor)
    return img

def part_2(
    img: Image,
    target_class: int,
    model: vgg19,
    device: str | torch.device
) -> Image:
    epsilon = 7.95/255
    iters = 100
    learning_rate = 0.001
    thresh = 0.85

    original_img = img2tensorVGG(img, device)
    img_tensor = original_img.clone().detach().requires_grad_(True)
    target_tensor = torch.tensor([target_class], device=device)

    for i in range(iters):
        img_tensor.grad = None
        outputs = model(img_tensor)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        target_prob = probs[0, target_class].item()

        if target_prob > thresh:
            break

        loss = torch.nn.functional.cross_entropy(outputs, target_tensor)

        loss.backward()

        with torch.no_grad():
            img_tensor = img_tensor - learning_rate * img_tensor.grad.sign()
            diff = img_tensor - original_img
            diff = torch.clamp(diff, -epsilon, epsilon)
            img_tensor = torch.clamp(original_img + diff, 0, 1).requires_grad_()

    img = tensor2imgVGG(img_tensor)
    return img

def part_3(
    img: Image,
    target_class: int,
    ensemble_model_1: vgg19,
    ensemble_model_2: vgg19,
    ensemble_model_3: vgg19,
    device: str | torch.device,
) -> Image:
    epsilon = 7.95/255
    iters = 500
    learning_rate = 3.5/255
    max_lr = 5/255
    min_lr = 0.25/255
    patience = 7
    patience_ctr = 0
    lr_decay = 0.85
    lr_incr = 1.05
    best_loss = float('inf')
    models = [ensemble_model_1, ensemble_model_2, ensemble_model_3]
    model_weights = [1/3, 1/3, 1/3]

    original_img = img2tensorVGG(img, device)
    img_tensor = original_img.clone().detach().requires_grad_(True)
    target_tensor = torch.tensor([target_class], device=device)

    for i in range(iters):
        img_tensor.grad = None
        loss = 0
        # if (i % 25 == 0):
        #     print(f"iter {i}: ", end="", flush=True)
        for m, w in zip(models, model_weights):
            outputs = m(img_tensor)
            loss_m = w * torch.nn.functional.cross_entropy(outputs, target_tensor)
            loss += loss_m
            # if (i % 25 == 0):
            #     print(loss_m, end="", flush=True)
        # if (i % 25 == 0):
        #     print(loss)
        loss_val = loss.item()
        if (loss_val < 0.0005):
            break

        # if loss_val < best_loss:
        #     patience_ctr = 0
        #     learning_rate = 3.5/255
        #     best_loss = loss_val
        # else:
        #     patience_ctr += 1
        #     if patience_ctr > patience:
        #         learning_rate = max(learning_rate * lr_decay, min_lr)
        #     if learning_rate == min_lr:
        #         print(':(', end="", flush=True)
        #         break
        # if i == 499:
        #     print('!', end="", flush=True)
        # it somehow got worse after i added all this :(((

        loss.backward()

        with torch.no_grad():
            img_tensor = img_tensor - learning_rate * img_tensor.grad.sign()
            diff = img_tensor - original_img
            diff = torch.clamp(diff, -epsilon, epsilon)
            img_tensor = torch.clamp(original_img + diff, 0, 1).requires_grad_()

    img = tensor2imgVGG(img_tensor)
    return img

def bonus(
    img: Image,
    target_class: int,
    endpoint_url: str,
    query_limit: int,
    device: str | torch.device
) -> Image:
    original_img = img2tensorVGG(img, device)
    img_tensor = original_img.clone().detach()

    epsilon = 11.75/255
    sigma = 0.0025
    n = 20
    learning_rate = 0.01

    thresh = 0.95

    best_prob = 0.0
    best_img_tensor = img_tensor.clone()

    num_iterations = query_limit // (2 * n)

    for i in range(num_iterations):
        grad_estimate = torch.zeros_like(img_tensor)


        probs = []

        for j in range(n):
            noise = torch.randn_like(img_tensor)

            tensor_pos = img_tensor + sigma * noise
            img_pos = tensor2imgVGG(tensor_pos)

            img_pos_byte_arr = io.BytesIO()
            img_pos.save(img_pos_byte_arr, format='PNG')
            img_pos_byte_arr.seek(0)
            files_pos = {"file": ("image.png", img_pos_byte_arr, "image/png")}
            response_pos = requests.post(endpoint_url, files=files_pos)
            outputs_pos = torch.tensor(response_pos.json()["output"])
            probs_pos = torch.nn.functional.softmax(outputs_pos, dim=0)

            probs.append(probs_pos[target_class].item())
            grad_estimate += probs_pos[target_class] * noise


            tensor_neg = img_tensor - sigma * noise
            img_neg = tensor2imgVGG(tensor_neg)

            img_neg_byte_arr = io.BytesIO()
            img_neg.save(img_neg_byte_arr, format='PNG')
            img_neg_byte_arr.seek(0)
            files_neg = {"file": ("image.png", img_neg_byte_arr, "image/png")}
            response_neg = requests.post(endpoint_url, files=files_neg)
            outputs_neg = torch.tensor(response_neg.json()["output"])
            probs_neg = torch.nn.functional.softmax(outputs_neg, dim=0)

            probs.append(probs_neg[target_class].item())
            grad_estimate -= probs_neg[target_class] * noise
        curr_max_prob = max(probs)
        if curr_max_prob > best_prob:
            best_prob = curr_max_prob
            best_img_tensor = img_tensor.clone()

        grad_estimate = grad_estimate / (2 * n * sigma)

        img_tensor = img_tensor + learning_rate * grad_estimate.sign()
        diff = img_tensor - original_img
        diff = torch.clamp(diff, -epsilon, epsilon)
        img_tensor = torch.clamp(original_img + diff, 0, 1)

        if curr_max_prob > thresh:
            break
    img = tensor2imgVGG(best_img_tensor)
    return img