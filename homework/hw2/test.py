"""
Use this file to test your HW2 solution.
Modify it, swap in different source images and targets, etc.

We will NOT be releasing the exact source-target pairs we will be grading,
but we will be testing your solutions in a manner very similar
to what is described below.
"""

from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import requests
import io

from hw2_starter import part_1, part_2, part_3, bonus

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def get_image_diff(src, adv):
    src_ten = torchvision.transforms.ToTensor()(src)
    adv_ten = torchvision.transforms.ToTensor()(adv)
    diff = torch.abs(adv_ten - src_ten)
    max_diff = diff.max().item() * 255
    if max_diff > 8.01:
        print(diff.max().item() * 255)
        return False
    return True

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# source and target for testing
source_img, source_class = trainset[6] # frog
target_class = 5 # dog

# >>> TESTING PART 1 >>>

# set up resnet model
resnet_model = ResNet18()
resnet_model.to(device)
resnet_model.load_state_dict(torch.load("./models/resnet18.pth", map_location=torch.device(device), weights_only=True))
resnet_model.eval()

# get adversarial image
adv_img = part_1(source_img, target_class, resnet_model, device)

# test adversarial image
with torch.no_grad():
    adv_tensor = img2tensorResNet(adv_img, device)
    output = resnet_model(adv_tensor)
    _, predicted_class = torch.max(output, 1)
if not get_image_diff(source_img, adv_img):
    print("\033[91mD\033[0m", end="", flush=True)
elif predicted_class == target_class:
    print("Part 1 worked")
else:
    print("Part 1 did not work")

# <<< END TESTING PART 1 <<<

# >>> TESTING PART 2 >>>

# set up trapdoor model
trapdoor_vgg_model = vgg19()
trapdoor_vgg_model.to(device)
trapdoor_vgg_model.load_state_dict(torch.load("./models/trapdoor_vgg.pth", map_location=torch.device(device), weights_only=True))
trapdoor_vgg_model.eval()

# set up detector
loaded_threshold_data = torch.load("./models/thresholds.pt", weights_only=False)
thresholds = {label: threshold for threshold, label in zip(loaded_threshold_data["thresholds"], loaded_threshold_data["labels"])}

loaded_signature_data = torch.load("./models/signatures.pt", weights_only=False)
signatures = {label: signature.to(device) for signature, label in zip(loaded_signature_data["signatures"], loaded_signature_data["labels"])}

feature_extractor = nn.Sequential(*list(trapdoor_vgg_model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

def attack_detected(img: Image, device):
    """
    returns True if detected as an attack
    returns False if no attack is detected
    """

    with torch.no_grad():
        img_tensor = img2tensorVGG(img, device)

        output = trapdoor_vgg_model(img_tensor)
        class_to_check = torch.max(output, 1)[1].item()

        signature = signatures[class_to_check]
        threshold = thresholds[class_to_check]

        features = feature_extractor(img_tensor)

        cos_sim = F.cosine_similarity(
            signature.flatten().unsqueeze(0),
            features.flatten().unsqueeze(0)
        ).item()

        return cos_sim > threshold

# get adversarial image
adv_img = part_2(source_img, target_class, trapdoor_vgg_model, device)

# test adversarial image
with torch.no_grad():
    adv_tensor = img2tensorVGG(adv_img, device)
    output = trapdoor_vgg_model(adv_tensor)
    _, predicted_class = torch.max(output, 1)

evaded_detection = not attack_detected(adv_img, device)

if not get_image_diff(source_img, adv_img):
    print("\033[91mD\033[0m", end="", flush=True)
elif predicted_class == target_class and evaded_detection:
    print("Part 2 attack worked")
elif predicted_class == target_class:
    print("Part 2 attack succeeded, but was detected")
elif evaded_detection:
    print("Part 2 evaded detection, but the attack was unsuccessful")
else:
    print("Part 2 attack did not work")

# <<< END TESTING PART 2 <<<

# >>> TESTING PART 3 >>>

# loading model ensemble
ensemble_1 = vgg19()
ensemble_1.to(device)
ensemble_1.load_state_dict(torch.load("./models/ensemble_1.pth", map_location=torch.device(device), weights_only=True))
ensemble_1.eval()

ensemble_2 = vgg19()
ensemble_2.to(device)
ensemble_2.load_state_dict(torch.load("./models/ensemble_2.pth", map_location=torch.device(device), weights_only=True))
ensemble_2.eval()

ensemble_3 = vgg19()
ensemble_3.to(device)
ensemble_3.load_state_dict(torch.load("./models/ensemble_3.pth", map_location=torch.device(device), weights_only=True))
ensemble_3.eval()

# get adversarial image
adv_img = part_3(
    source_img,
    target_class,
    ensemble_1,
    ensemble_2,
    ensemble_3,
    device
)

# test output
img_byte_arr = io.BytesIO()
adv_img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)
files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
response = requests.post("http://floo.cs.uchicago.edu/hw2_black_box", files=files)

if not get_image_diff(source_img, adv_img):
    print("\033[91mD\033[0m", end="", flush=True)
elif response.json()["class"] == target_class:
    print("Part 3 worked")
else:
    print("Part 3 did not work")

# <<< END TESTING PART 3 <<<

# >>> TESTING BONUS >>>

endpoint_url = "http://floo.cs.uchicago.edu/hw2_black_box"

adv_img = bonus(
    source_img,
    target_class,
    endpoint_url,
    7000,
    device
)

# test output
img_byte_arr = io.BytesIO()
adv_img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)
files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
response = requests.post(endpoint_url, files=files)

if not get_image_diff(source_img, adv_img):
    print("\033[91mD\033[0m", end="", flush=True)
elif response.json()["class"] == target_class:
    print("Bonus worked")
else:
    print("Bonus did not work")

# <<< END TESTING BONUS
