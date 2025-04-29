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

import kornia.augmentation as K

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class State:
    def __init__(self):
        self.model = None
        self.test_loader = None
        self.train_loader = None
        self.num_samples = None
        self.question = None
    def set_model(self, model):
        self.model = model
    def set_test_loader(self, test_loader):
        self.test_loader = test_loader
    def set_train_loader(self, train_loader):
        self.train_loader = train_loader
    def set_num_samples(self, num_samples):
        self.num_samples = num_samples
    def set_question(self, question):
        self.question = question

state = State()
# --------- Part 1: Simple Transformations + Evaluation ---------

def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Applies JPEG compression to the input image tensor
    """
    img = tensor2imgVGG(x)
    jpeg = v2.JPEG(50)
    compressed_img = jpeg(img)
    ten = img2tensorVGG(compressed_img, device)
    return ten


def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies resizing and rescaling to the input image tensor
    """
    img = tensor2imgVGG(x)
    resize = v2.Resize(16)
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
    blur = v2.GaussianBlur(kernel_size=5, sigma=1.75)
    blurred_img = blur(img)
    ten = img2tensorVGG(blurred_img, device)
    return ten

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    samples = []
    sample_indices = torch.randperm(len(state.test_loader.dataset))[:state.num_samples].tolist()

    for idx in sample_indices:
        image, label = state.test_loader.dataset[idx]
        possible_targets = list(range(10))
        possible_targets.remove(label)
        ae_label = possible_targets[torch.randint(0, len(possible_targets), (1,)).item()]
        samples.append((image, label, ae_label))

    state.model.eval()

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
            ae = img2tensorVGG(target_pgd_attack(tensor2imgVGG(image), ae_label, state.model, device), device)
        else:
            ae = img2tensorVGG(eot_attack(state.model, image, torch.tensor(ae_label).unsqueeze(0)), device)

        image = image.unsqueeze(0)

        img_comped = jpeg_compression(image)
        img_resized = image_resizing(image)
        img_blurred = gaussian_blur(image)
        ae_comped = jpeg_compression(ae)
        ae_resized = image_resizing(ae)
        ae_blurred = gaussian_blur(ae)

        with torch.no_grad():
            benign_class = state.model(image).argmax(dim=1).item()
            ae_class = state.model(ae).argmax(dim=1).item()
            img_comped_class = state.model(img_comped).argmax(dim=1).item()
            ae_comped_class = state.model(ae_comped).argmax(dim=1).item()
            img_resized_class = state.model(img_resized).argmax(dim=1).item()
            ae_resized_class = state.model(ae_resized).argmax(dim=1).item()
            img_blurred_class = state.model(img_blurred).argmax(dim=1).item()
            ae_blurred_class = state.model(ae_blurred).argmax(dim=1).item()

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

    comp = K.RandomJPEG(jpeg_quality=50)
    resizing = K.RandomResizedCrop(size=(32, 32), scale=(0.25, 0.25), ratio=(1.0, 1.0), p=1.0)
    blur = K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(1.75, 1.75), p=1.0)

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

    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in state.train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
            soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=1)


            # loss = -(torch.sum(soft_targets * soft_prob)) / soft_prob.size()[0]
            # loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
            # label_loss = nn.functional.cross_entropy(student_logits, labels)
            distillation_loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (temperature * temperature)
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            loss = 0.25 * distillation_loss + 0.75 * hard_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(state.train_loader)}")

    torch.save(student.state_dict(), "models/student_VGG.pth")
# loss = (torch.sum(soft_targets * (soft_targets.log() - soft_prob))
            #                         / soft_prob.size()[0] * (temperature ** 2))
def evaluate_distillation():
    """
    Evaluates the student model on clean data and under targeted PGD attack
    """
    samples = []
    sample_indices = torch.randperm(len(state.test_loader.dataset))[:state.num_samples].tolist()

    for idx in sample_indices:
        image, label = state.test_loader.dataset[idx]
        possible_targets = list(range(10))
        possible_targets.remove(label)
        ae_label = possible_targets[torch.randint(0, len(possible_targets), (1,)).item()]
        samples.append((image, label, ae_label))

    state.model.eval()

    clean_correct = 0
    ae_correct = 0
    ae_success = 0

    for image, label, ae_label in samples:
        print(".", end="", flush=True)
        image = image.to(device)
        ae = img2tensorVGG(target_pgd_attack(tensor2imgVGG(image), ae_label, state.model, device), device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            clean_class = state.model(image).argmax(dim=1).item()
            ae_class = state.model(ae).argmax(dim=1).item()

        if clean_class == label:
            clean_correct += 1
        if ae_class == label:
            ae_correct += 1
        elif ae_class == ae_label:
            ae_success += 1

    print("\n")
    clean_correct /= (state.num_samples / 100)
    ae_correct /= (state.num_samples / 100)
    ae_success /= (state.num_samples / 100)

    return {
        "base": [clean_correct, ae_correct, ae_success, ],
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

def output_results(res, part):
    print(f"\n----- PART {part}: Transformations -----")
    print("-" * 60)
    print(f"{'Defense':<10} | {'Clean Acc':<10} | {'AE Acc':<10} | {'Attack Success':<15}")
    print("-" * 60)

    for defense, metrics in res.items():
        clean_acc, ae_robust, attack_success = metrics
        defense_name = {"base": "None", "comp": "JPEG", "resized": "Resize", "blur": "Blur"}[defense]
        print(
            f"{defense_name:<10} | {clean_acc:.2f}%{' ':<5} | {ae_robust:.2f}%{' ':<5} | {attack_success:.2f}%{' ':<10}")

def main():
    # load data
    train_loader, test_loader = load_dataset()

    # use gpu device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # load teacher model
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))

    state.model = model
    state.test_loader = test_loader
    state.train_loader = train_loader
    state.num_samples = 1
    state.question = 1

    # PART 1: Evaluate simple defenses

    results = evaluate_transformations()

    output_results(results, 1)

    # PART 2: EOT Attack
    state.num_samples = 1
    state.question = 2

    results_eot = evaluate_transformations()

    output_results(results_eot, 2)

    # PART 3: Distillation Defense
    student_VGG(temperature=1000)
    student = VGG('VGG16').to(device)
    student.load_state_dict(torch.load("models/student_VGG.pth", map_location=device))

    state.model = student
    state.num_samples = 100
    state.question = 3

    results_dist = evaluate_distillation()

    output_results(results_dist, 3)

if __name__ == "__main__":
    main()
