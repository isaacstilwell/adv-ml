# hw3_starter.py
from symtable import Class

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

# import class functions:
import hw3_utils
from hw3_utils import target_pgd_attack, tensor2imgVGG, img2tensorVGG
from model import VGG, load_dataset

import kornia.augmentation as KA
from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
import kornia.geometry as KG
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import jpeg_codec_differentiable

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class State:
    def __init__(self):
        self.teacher = None
        self.student = None
        self.test_loader = None
        self.train_loader = None
        self.num_samples = None
        self.question = None
        self.eval_set = None

""" 
    NOTE TO CS 280 GRADER FROM ISAAC: This patch and RandomJPEG are from kornia. 
    I had to fix thee codec_differentiable function locally to work with CUDA referencing this PR:
    https://github.com/kornia/kornia/pull/2883
"""

def patched_jpeg_codec_differentiable(image_rgb, jpeg_quality, **kwargs):
    d = image_rgb.device
    jpeg_quality = jpeg_quality.to(d)

    return jpeg_codec_differentiable(image_rgb, jpeg_quality, **kwargs)

class RandomJPEG(IntensityAugmentationBase2D):
    def __init__(
        self,
        jpeg_quality: Union[Tensor, float, Tuple[float, float], List[float]] = 50.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.jpeg_quality = jpeg_quality
        self._param_generator = rg.JPEGGenerator(jpeg_quality)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        jpeg_output: Tensor = jpeg_codec_differentiable(input, params["jpeg_quality"])
        return jpeg_output


state = State()
# --------- Part 1: Simple Transformations + Evaluation ---------

def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Applies JPEG compression to the input image tensor
    """
    img = tensor2imgVGG(x)
    jpeg = v2.JPEG(28) #32
    compressed_img = jpeg(img)
    ten = img2tensorVGG(compressed_img, device)
    return ten


def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies resizing and rescaling to the input image tensor
    """
    img = tensor2imgVGG(x)
    resize = v2.Resize(13) #13
    resized_img = resize(img)

    restore = v2.Resize(32)
    restored_img = restore(resized_img)

    ten = img2tensorVGG(restored_img, device)
    return ten

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian blur to the input image tensor
    """
    img = tensor2imgVGG(x)
    blur = v2.GaussianBlur(kernel_size=5, sigma=2)
    blurred_img = blur(img)
    ten = img2tensorVGG(blurred_img, device)
    return ten

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    samples = []

    # randomly generated ae_labels from commented ae_label code below; saved for consistency in evaluating parameters
    ae_labels = [7, 5, 4, 4, 7, 0, 8, 5, 9, 9, 1, 3, 9, 4, 8, 6, 9, 8, 2, 5, 1, 7, 3, 5, 0, 5, 5, 1, 2, 1, 4, 7, 7, 8,
                 1, 1, 1, 8, 1, 4, 2, 8, 9, 2, 1, 1, 4, 7, 5, 8, 4, 0, 8, 6, 4, 1, 9, 0, 5, 0, 5, 0, 4, 5, 5, 6, 9, 9,
                 9, 1, 6, 6, 8, 3, 8, 1, 3, 3, 6, 2, 0, 2, 6, 8, 8, 9, 0, 1, 2, 5, 1, 0, 9, 1, 0, 4, 8, 6, 4, 0]

    for meta_idx, idx in enumerate(state.eval_set):
        image, label = state.test_loader.dataset[idx]
        possible_targets = list(range(10))
        possible_targets.remove(label)
        # ae_label = possible_targets[torch.randint(0, len(possible_targets), (1,)).item()]
        ae_label = ae_labels[meta_idx]
        samples.append((image, label, ae_label))

    state.teacher.eval()

    base_benign_correct = 0
    base_ae_correct = 0
    base_ae_success = 0

    comp_benign_correct = 0
    comp_ae_correct = 0
    comp_ae_success = 0

    resized_benign_correct = 0
    resized_ae_correct = 0
    resized_ae_success = 0

    blur_benign_correct = 0
    blur_ae_correct = 0
    blur_ae_success = 0


    for image, label, ae_label in samples:
        print(".", end="", flush=True)
        image = image.to(device)
        if state.question == 1:
            ae = img2tensorVGG(target_pgd_attack(tensor2imgVGG(image), ae_label, state.teacher, device), device)
        else:
            ae = img2tensorVGG(eot_attack(state.teacher, image, torch.tensor(ae_label).unsqueeze(0)), device)

        image = image.unsqueeze(0)

        img_comped = jpeg_compression(image)
        img_resized = image_resizing(image)
        img_blurred = gaussian_blur(image)
        ae_comped = jpeg_compression(ae)
        ae_resized = image_resizing(ae)
        ae_blurred = gaussian_blur(ae)

        with torch.no_grad():
            benign_class = state.teacher(image).argmax(dim=1).item()
            ae_class = state.teacher(ae).argmax(dim=1).item()
            img_comped_class = state.teacher(img_comped).argmax(dim=1).item()
            ae_comped_class = state.teacher(ae_comped).argmax(dim=1).item()
            img_resized_class = state.teacher(img_resized).argmax(dim=1).item()
            ae_resized_class = state.teacher(ae_resized).argmax(dim=1).item()
            img_blurred_class = state.teacher(img_blurred).argmax(dim=1).item()
            ae_blurred_class = state.teacher(ae_blurred).argmax(dim=1).item()

        if benign_class == label:
            base_benign_correct += 1
        if ae_class == label:
            base_ae_correct += 1
        elif ae_class == ae_label:
            base_ae_success += 1

        if img_comped_class == label:
            comp_benign_correct += 1
        if ae_comped_class == label:
            comp_ae_correct += 1
        elif ae_comped_class == ae_label:
            comp_ae_success += 1

        if img_resized_class == label:
            resized_benign_correct += 1
        if ae_resized_class == label:
            resized_ae_correct += 1
        elif ae_resized_class == ae_label:
            resized_ae_success += 1

        if img_blurred_class == label:
            blur_benign_correct += 1
        if ae_blurred_class == label:
            blur_ae_correct += 1
        elif ae_blurred_class == ae_label:
            blur_ae_success += 1
    print("\n")
    base_benign_correct /= (state.num_samples / 100)
    base_ae_correct /= (state.num_samples / 100)
    base_ae_success /= (state.num_samples / 100)

    comp_benign_correct /= (state.num_samples / 100)
    comp_ae_correct /= (state.num_samples / 100)
    comp_ae_success /= (state.num_samples / 100)

    resized_benign_correct /= (state.num_samples / 100)
    resized_ae_correct /= (state.num_samples / 100)
    resized_ae_success /= (state.num_samples / 100)

    blur_benign_correct /= (state.num_samples / 100)
    blur_ae_correct /= (state.num_samples / 100)
    blur_ae_success /= (state.num_samples / 100)

    return {
        "base": [base_benign_correct, base_ae_correct, base_ae_success,],
        "comp": [comp_benign_correct, comp_ae_correct, comp_ae_success,],
        "resized": [resized_benign_correct, resized_ae_correct, resized_ae_success,],
        "blur": [blur_benign_correct, blur_ae_correct, blur_ae_success,],
    }

# --------- Part 2: EOT Attack + Evaluation ---------

def eot_attack(model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        model: The target model to attack
        x: Input image (clean)
        y_target: Target label 
    
    Returns:
        Adversarial example
    """
    epsilon = 8 / 255
    tensor_max = 1
    tensor_min = 0
    lr_initial = 0.01
    max_iter = 200

    modifier = torch.zeros_like(x, requires_grad=True)

    target_label = y_target.to(device)
    loss_fn = nn.CrossEntropyLoss()

    comp = RandomJPEG(jpeg_quality=35)
    # resizing = KA.RandomResizedCrop(size=(32, 32), scale=(0.25, 0.25), ratio=(1.0, 1.0), p=1.0)
    resizing = KA.AugmentationSequential(
        KA.Resize((13, 13)),
        KA.Resize((32, 32)),
        same_on_batch=True
    )
    blur = KA.RandomGaussianBlur(kernel_size=(5, 5), sigma=(2, 2), p=1.0)

    trans = [comp, resizing, blur]

    for i in range(max_iter):
        adv_tensor = torch.clamp(x + modifier, tensor_min, tensor_max)
        loss = 0
        outputs = []
        for t in trans:
            output = model(t(adv_tensor))
            loss += loss_fn(output, target_label)
            outputs.append(output)
        loss /= len(trans)

        model.zero_grad()
        if modifier.grad is not None:
            modifier.grad.zero_()
        loss.backward()

        grad = modifier.grad
        modifier = modifier - lr_initial * grad.sign()
        modifier = torch.clamp(modifier, min=-epsilon, max=epsilon).detach().requires_grad_(True)

        if i % (max_iter // 10) == 0:
            num_correct = 0
            for o in outputs:
                pred_class = torch.argmax(o, dim=1).item()
                if pred_class == y_target:
                    num_correct += 1

            # Optional: uncomment to print loss values:
            # print(f"step: {i} | loss: {loss.item():.4f} | pred class: {classes[pred_class]}")

            if num_correct == len(outputs):
                break

    adv_tensor = torch.clamp(x + modifier, tensor_min, tensor_max)
    return tensor2imgVGG(adv_tensor)

# --------- Part 3: Defensive Distillation + Evaluation ---------

def student_VGG(teacher_path: str = "models/vgg16_cifar10_robust.pth", temperature: float = 20.0,) -> None:
    """
    Trains a student model using knowledge distillation and saves it as 'student_VGG.pth'

    Args:
        train_loader: The training data loader
        teacher_path: Path to the pretrained teacher model.
        temperature: Softmax temperature for distillation.
    """
    student = hw3_utils.get_vgg_model().to(device)
    student.train()

    teacher = VGG('VGG16').to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    num_epochs = 30
    for epoch in range(num_epochs):
        running_loss = 0.0
        idx = 0
        for images, labels in state.train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            # print("softmax: ", softmax[0])
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
            # if idx % 64 == 0:
            #     print(f"\nstudent prob: {prob[0].detach().cpu().tolist()}\n"
            #           f"teacher soft targets: {soft_targets[0].detach().cpu().tolist()}\n"
            #           f"teacher prob: {softmax[0].detach().cpu().tolist()}\n")
            # print("softtargets: ", soft_targets[0])
            soft_prob = nn.functional.log_softmax(student_logits, dim=1)

            stu_prob = nn.functional.softmax(student_logits, dim=1)


            # loss = -(torch.sum(soft_targets * soft_prob)) / soft_prob.size()[0]
            # loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
            # label_loss = nn.functional.cross_entropy(student_logits, labels)
            loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (temperature * temperature)
            # hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            # loss = nn.functional.cross_entropy(stu_prob, soft_targets)
            # loss = 0.25 * distillation_loss + 0.75 * hard_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            idx += 1

        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(state.train_loader)}")
        if running_loss / len(state.train_loader) < 0.7:
            break

    torch.save(student.state_dict(), "models/student_VGG.pth")
# loss = (torch.sum(soft_targets * (soft_targets.log() - soft_prob))
            #                         / soft_prob.size()[0] * (temperature ** 2))
def evaluate_distillation():
    """
    Evaluates the student model on clean data and under targeted PGD attack
    """
    samples = []

    # randomly generated ae_labels from commented ae_label code below; saved for consistency in evaluating parameters
    ae_labels = [7, 5, 4, 4, 7, 0, 8, 5, 9, 9, 1, 3, 9, 4, 8, 6, 9, 8, 2, 5, 1, 7, 3, 5, 0, 5, 5, 1, 2, 1, 4, 7, 7, 8,
                 1, 1, 1, 8, 1, 4, 2, 8, 9, 2, 1, 1, 4, 7, 5, 8, 4, 0, 8, 6, 4, 1, 9, 0, 5, 0, 5, 0, 4, 5, 5, 6, 9, 9,
                 9, 1, 6, 6, 8, 3, 8, 1, 3, 3, 6, 2, 0, 2, 6, 8, 8, 9, 0, 1, 2, 5, 1, 0, 9, 1, 0, 4, 8, 6, 4, 0]

    for meta_idx, idx in enumerate(state.eval_set):
        image, label = state.test_loader.dataset[idx]
        possible_targets = list(range(10))
        possible_targets.remove(label)
        # ae_label = possible_targets[torch.randint(0, len(possible_targets), (1,)).item()]
        ae_label = ae_labels[meta_idx]
        samples.append((image, label, ae_label))
    print("[", end="", flush=True)
    for _, l, a in samples:
        print(f"({l, a}),", end="", flush=True)
    print("]", end="", flush=True)
    print("")

    state.student.eval()

    teacher_correct = 0
    clean_correct = 0
    ae_correct = 0
    ae_success = 0

    for i, (image, label, ae_label) in enumerate(samples):
        image = image.to(device)
        ae = img2tensorVGG(target_pgd_attack(tensor2imgVGG(image), ae_label, state.student, device), device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            if i == 0:
                student_image_softmax = nn.functional.softmax(state.student(image))
                student_ae_softmax = nn.functional.softmax(state.student(ae))
                teacher_image_softmax = nn.functional.softmax(state.teacher(image))
                teacher_ae_softmax = nn.functional.softmax(state.teacher(ae))

                print(f"stu_image_softmax: {student_image_softmax}\nteacher_image_softmax: {teacher_image_softmax}\nstu_ae_softmax: {student_ae_softmax}\nteacher_ae_softmax: {teacher_ae_softmax}")
            teacher_clean = state.teacher(image).argmax(dim=1).item()
            clean_class = state.student(image).argmax(dim=1).item()
            ae_class = state.student(ae).argmax(dim=1).item()

        print(".", end="", flush=True)

        if clean_class == label:
            clean_correct += 1
        if teacher_clean == label:
            teacher_correct += 1
        if ae_class == label:
            ae_correct += 1
        elif ae_class == ae_label:
            ae_success += 1

    print("\n")
    clean_correct /= (state.num_samples / 100)
    teacher_correct /= (state.num_samples / 100)
    ae_correct /= (state.num_samples / 100)
    ae_success /= (state.num_samples / 100)

    return {
        "dist": [clean_correct, teacher_correct, ae_correct, ae_success, ],
    }

# --------- Bonus Part: Adaptive Attack ---------

def adaptive_attack(student_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """
    Bonus: Implements a stronger adaptive attack on distilled student model from Part 3

    Args:
        student_model: The distilled student model to attack
        x: Clean input images
        y_target: Target labels

    Returns:
        Adversarial examples
    """
    pass

def create_eval_set(size):
    if size % 10 != 0:
        raise ValueError("size must be divisible by 10")
    counts = {}
    indices = []
    skip = []
    print(len(state.test_loader.dataset))
    for idx, (_, targ) in enumerate(state.test_loader.dataset):
        if targ in skip:
            continue
        counts[targ] = counts.get(targ, 0) + 1
        indices.append(idx)
        if counts[targ] == 10:
            skip.append(targ)

        if len(indices) == size:
            break
    print(counts)

    return indices

def output_results(res, part):
    print(f"\n----- PART {part}: Transformations -----")
    print("-" * 60)
    if part != 3:
        print(f"{'Defense':<10} | {'Clean Acc':<10} | {'AE Acc':<10} | {'Attack Success':<15}")
    else:
        print(f"{'Defense':<10} | {'Student Clean Acc':<20} | {'Teacher Clean Acc':<20} {'Student AE Acc':<15} | {'Attack Success':<15}")
    print("-" * 60)

    for defense, metrics in res.items():
        defense_name = {"base": "None", "comp": "JPEG", "resized": "Resize", "blur": "Blur", "dist": "Distillation"}[defense]
        if part != 3:
            clean_acc, ae_robust, attack_success = metrics
            print(
                f"{defense_name:<10} | {clean_acc:.2f}%{' ':<5} | {ae_robust:.2f}%{' ':<5} | {attack_success:.2f}%{' ':<10}")
        else:
            clean_acc, teacher_acc, ae_robust, attack_success = metrics
            print(
                f"{defense_name:<10} | {clean_acc:.2f}%{' ':<5} | {teacher_acc:.2f}%{' ':<5} | {ae_robust:.2f}%{' ':<5} | {attack_success:.2f}%{' ':<10}")

def main():
    # load data
    train_loader, test_loader = load_dataset()
    num_samples = 100

    # use gpu device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # load teacher model
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))

    state.teacher = model
    state.test_loader = test_loader
    state.train_loader = train_loader
    state.num_samples = num_samples
    state.question = 1
    state.eval_set = create_eval_set(num_samples)
    print(state.eval_set)

    # PART 1: Evaluate simple defenses

    results = evaluate_transformations()

    output_results(results, 1)

    # PART 2: EOT Attack
    state.num_samples = num_samples
    state.question = 2

    results_eot = evaluate_transformations()

    output_results(results_eot, 2)

    # PART 3: Distillation Defense
    # for i in range(75):

    # student_VGG(temperature=90) # 90, 95
    student = VGG('VGG16').to(device)
    student.load_state_dict(torch.load("models/student_VGG.pth", map_location=device))

    state.student = student
    state.num_samples = num_samples
    state.question = 3

    results_dist = evaluate_distillation()
    output_results(results_dist, 3)

if __name__ == "__main__":
    main()
