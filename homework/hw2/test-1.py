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
import random
import requests
import io

from hw2_starter import part_1, part_2, part_3, bonus

if torch.cuda.is_available():
    print('using cuda')
    device = "cuda"
elif torch.mps.is_available():
    print('using mps')
    device = "mps"
else:
    print('using cpu')
    device = "cpu"

def get_image_diff(src, adv):
    src_ten = torchvision.transforms.ToTensor()(src)
    adv_ten = torchvision.transforms.ToTensor()(adv)
    diff = torch.abs(adv_ten - src_ten)
    max_diff = diff.max().item() * 255
    if max_diff > 8.01:
        print(diff)
        return False
    return True


# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32

NUM_ITERS = 1

a_worked = 0
a_failed = 0
a_failures = []

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

print("p1 tests: ", end="", flush=True)
for a in range(0, NUM_ITERS):
    idx = random.randint(0, len(trainset) - 1)
# source and target for testing
    source_img, source_class = trainset[idx] # frog
    source_img.save('source.jpg')
    target_class = random.randint(0, 9)
    while (target_class == source_class):
        target_class = random.randint(0, 9)

    # >>> TESTING PART 1 >>>

    # set up resnet model
    resnet_model = ResNet18()
    resnet_model.to(device)
    resnet_model.load_state_dict(torch.load("./models/resnet18.pth", map_location=torch.device(device), weights_only=True))
    resnet_model.eval()

    # get adversarial image
    adv_img = part_1(source_img, target_class, resnet_model, device)
    # adv_img.save('adv.jpg')

    # test adversarial image
    with torch.no_grad():
        adv_tensor = img2tensorResNet(adv_img, device)
        output = resnet_model(adv_tensor)
        _, predicted_class = torch.max(output, 1)

    if not get_image_diff(source_img, adv_img):
        print("\033[91mD\033[0m", end="", flush=True)
        a_failed += 1
        a_failures.append((idx, "diff."))
    elif predicted_class == target_class:
        print("\033[92m.\033[0m", end="", flush=True)
        a_worked += 1
    else:
        print("\033[91mX\033[0m", end="", flush=True)
        a_failed += 1
        a_failures.append((idx, source_class))
print("")
print(a_worked + a_failed, a_worked, a_failed)
print(a_failures)

b_worked = 0
b_failed = 0
b_failures = []

print("p2 tests: ", end="", flush=True)
for a in range(0, NUM_ITERS):
    idx = random.randint(0, len(trainset) - 1)
    # source and target for testing
    source_img, source_class = trainset[idx] # frog
    target_class = random.randint(0, 9)
    while (target_class == source_class):
        target_class = random.randint(0, 9)
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
        b_failed += 1
        b_failures.append((idx, "diff."))
    elif predicted_class == target_class and evaded_detection:
        print("\033[92m.\033[0m", end="", flush=True)
        b_worked += 1
    elif predicted_class == target_class:
        print("\033[93mD\033[0m", end="", flush=True)
        b_failed += 1
        b_failures.append((idx, source_class))
    elif evaded_detection:
        print("\033[93mF\033[0m", end="", flush=True)
        b_failed += 1
        b_failures.append((idx, source_class))
    else:
        print("\033[91mX\033[0m", end="", flush=True)
        b_failed += 1
        b_failures.append((idx, source_class))
print("")
print(b_worked + b_failed, b_worked, b_failed)
print(b_failures)

### ORIGINAL CODE ####
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# # source and target for testing
# source_img, source_class = trainset[6] # frog
# target_class = 5 # dog

# # >>> TESTING PART 1 >>>

# # set up resnet model
# resnet_model = ResNet18()
# resnet_model.to(device)
# resnet_model.load_state_dict(torch.load("./models/resnet18.pth", map_location=torch.device(device), weights_only=True))
# resnet_model.eval()

# # get adversarial image
# adv_img = part_1(source_img, target_class, resnet_model, device)

# # test adversarial image
# with torch.no_grad():
#     adv_tensor = img2tensorResNet(adv_img, device)
#     output = resnet_model(adv_tensor)
#     _, predicted_class = torch.max(output, 1)

# if predicted_class == target_class:
#     print("Part 1 worked")
# else:
#     print("Part 1 did not work")

# <<< END TESTING PART 1 <<<

# >>> TESTING PART 2 >>>

# set up trapdoor model
# trapdoor_vgg_model = vgg19()
# trapdoor_vgg_model.to(device)
# trapdoor_vgg_model.load_state_dict(torch.load("./models/trapdoor_vgg.pth", map_location=torch.device(device), weights_only=True))
# trapdoor_vgg_model.eval()

# # set up detector
# loaded_threshold_data = torch.load("./models/thresholds.pt", weights_only=False)
# thresholds = {label: threshold for threshold, label in zip(loaded_threshold_data["thresholds"], loaded_threshold_data["labels"])}

# loaded_signature_data = torch.load("./models/signatures.pt", weights_only=False)
# signatures = {label: signature.to(device) for signature, label in zip(loaded_signature_data["signatures"], loaded_signature_data["labels"])}

# feature_extractor = nn.Sequential(*list(trapdoor_vgg_model.children())[:-1])
# feature_extractor.to(device)
# feature_extractor.eval()

# def attack_detected(img: Image, device):
#     """
#     returns True if detected as an attack
#     returns False if no attack is detected
#     """

#     with torch.no_grad():
#         img_tensor = img2tensorVGG(img, device)

#         output = trapdoor_vgg_model(img_tensor)
#         class_to_check = torch.max(output, 1)[1].item()

#         signature = signatures[class_to_check]
#         threshold = thresholds[class_to_check]

#         features = feature_extractor(img_tensor)

#         cos_sim = F.cosine_similarity(
#             signature.flatten().unsqueeze(0),
#             features.flatten().unsqueeze(0)
#         ).item()

#         return cos_sim > threshold

# # get adversarial image
# adv_img = part_2(source_img, target_class, trapdoor_vgg_model, device)

# # test adversarial image
# with torch.no_grad():
#     adv_tensor = img2tensorVGG(adv_img, device)
#     output = trapdoor_vgg_model(adv_tensor)
#     _, predicted_class = torch.max(output, 1)

# evaded_detection = not attack_detected(adv_img, device)

# if predicted_class == target_class and evaded_detection:
#     print("Part 2 attack worked")
# elif predicted_class == target_class:
#     print("Part 2 attack succeeded, but was detected")
# elif evaded_detection:
#     print("Part 2 evaded detection, but the attack was unsuccessful")
# else:
#     print("Part 2 attack did not work")

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

# # get adversarial image
# adv_img = part_3(
#     source_img,
#     target_class,
#     ensemble_1,
#     ensemble_2,
#     ensemble_3,
#     device
# )

# # test output
# img_byte_arr = io.BytesIO()
# adv_img.save(img_byte_arr, format='PNG')
# img_byte_arr.seek(0)
# files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
# response = requests.post("http://floo.cs.uchicago.edu/hw2_black_box", files=files)

# if response.json()["class"] == target_class:
#     print("Part 3 worked")
# else:
#     print("Part 3 did not work")

c_worked = 0
c_failed = 0
c_failures = []



lr = [4/255, 3.9/255, 3.75/255, 3.66/255, 3.5/255, 3.33/255]
idxs = [37651, 32351, 33372, 26930, 32367, 17691, 36328, 34250, 669, 39405, 35211, 41431, 44576, 38438, 42096, 46781, 4957, 34170, 20640, 49342, 24649, 34576, 9333, 13139, 15725, 33224, 15584, 6680, 19345, 36244, 103, 8905, 25816, 37452, 9606, 1406, 2872, 23085, 32184, 25792, 4662, 28226, 11498, 12351, 41115, 27985, 205, 33314, 9036, 42355, 6978, 4063, 33624, 19051, 48756, 18650, 45144, 33022, 16267, 6134, 25658, 17895, 13834, 48693, 27302, 44858, 14805, 5789, 8531, 37590, 3833, 6067, 13597, 28580, 47172, 34420, 40626, 4260, 32466, 33315, 7462, 29590, 17370, 44915, 5865, 22055, 46644, 6915, 9369, 37977, 5191, 23627, 44755, 9547, 12000, 3648, 26432, 26174, 42568, 34333, 35147, 7655, 42208, 20014, 20672, 19501, 44557, 32355, 29281, 3699, 45632, 34477, 33498, 2521, 18986, 8180, 46306, 28591, 29548, 44606, 14091, 36188, 1030, 6122, 1065, 29615, 18832, 29290, 21483, 13311, 36378, 39438, 46948, 11135, 2936, 30346, 38987, 12939, 14139, 43221, 35114, 11628, 6981, 19329, 39276, 106, 41393, 6447, 43722, 33883, 503, 6300, 39733, 29197, 2051, 14485, 19176, 13399, 8746, 12524, 34439, 48869, 45175, 6056, 38411, 31413, 22558, 25795, 3691, 16152, 47071, 5001, 9877, 41072, 43529, 1062, 40595, 41286, 31478, 30951, 6238, 29224, 19926, 2800, 36720, 42981, 44032, 38916, 21463, 47648, 1018, 48560, 31374, 48712, 22979, 41690, 18605, 2227, 14546, 45896, 29861, 2710, 39347, 44229, 43809, 23296, 37503, 45257, 17024, 33208, 42088, 40550, 3003, 44585, 653, 38955, 22934, 43051, 43506, 6492, 18360, 2734, 2467, 11925, 32336, 37238, 22521, 32498, 35398, 10923, 22988, 16332, 31991, 45997, 14287, 29775, 13302, 38509, 47252, 44485, 24353, 12814, 7109, 7858, 7518, 34204, 2730, 24297, 23834, 20953, 31709, 35309, 12743, 20511, 489, 26801, 23818, 32925, 28839, 27850, 12865, 47884, 36249, 9312, 41475, 47867, 29148, 14879, 13983, 47950, 47121, 44515, 13298, 39522, 48990, 7936, 44803, 39158, 10352, 43990, 2841, 6034, 35340, 7667, 5768, 9054, 41755, 9006, 119, 7990, 15738, 37209, 38587, 22002, 9078, 2091, 35381, 11809, 10966, 44266, 49898, 15188, 16504, 20780, 41801, 7453, 48320, 28324, 8836, 48417, 39329, 17003, 37764, 10069, 33211, 17858, 11800, 37912, 14872, 38368, 10002, 12256, 21523, 39672, 10452, 49895, 8364, 49058, 27770, 31567, 24212, 25785, 31479, 48426, 9216, 1665, 14941, 9426, 38498, 30930, 7825, 19561, 12007, 49469, 43498, 14721, 41765, 44255, 27207, 27758, 30874, 10903, 1066, 26640, 46500, 38451, 47660, 32229, 14032, 43326, 46271, 18670, 43472, 1774, 46358, 5634, 30650, 46436, 14518, 27080, 13333, 34157, 35319, 29122, 3693, 46865, 32235, 38318, 2863, 36580, 21274, 7608, 38549, 38345, 1472, 9444, 49541, 14662, 31608, 41784, 42706, 25568, 17360, 29537, 36854, 49036, 37948, 27264, 13635, 22573, 16577, 30540, 12654, 44176, 22277, 16119, 3450, 21430, 49821, 48255, 20479, 42557, 43558, 5273, 2564, 31024, 16508, 11718, 15748, 7247, 12069, 33907, 45149, 25958, 48590, 7961, 6295, 9617, 24325, 37947, 5283, 13362, 9688, 38485, 31585, 18286, 42371, 33496, 45815, 10902, 11226, 23863, 26437, 1812, 12080, 17248, 6919, 32320, 19944, 22807, 41202, 5618, 36389, 7651, 15445, 30996, 47428, 23034, 25872, 26114, 49600, 5126, 25500, 1522, 37521, 11294, 33875, 7074, 33675, 3602, 26323, 19464, 48067, 45995, 41086, 39316, 8600, 9056, 20462, 36283, 42731, 9829, 14470, 21427, 36882, 16017, 1213, 21922, 26560, 33398, 17471, 1415, 14477, 18258, 30115, 38892, 41152, 41022, 21496, 33903]
iters = [1]#[100, 250, 500, 1000, 2000]
broken = [(1, 1)]#[(15804, 6), (10328, 7), (10420, 1), (49592, 7), (3257, 1), (25007, 1), (45302, 9), (49491, 1), (9121, 0), (18641, 7), (43758, 4), (43439, 6), (48442, 9), (38179, 0), (13902, 0), (24406, 7), (29130, 9), (16087, 1), (28416, 6), (46287, 0), (23789, 1), (11075, 1), (45631, 3), (13565, 8), (30208, 1), (17628, 8), (27719, 9), (45449, 7), (20635, 9), (10164, 8), (37438, 4), (46268, 1), (19937, 8), (43635, 9)]
for b in broken:
    print(f"p3 test {b[0]}: ", end="\n", flush=True)
    # idxs = []
    # for i in range(0, 500):
    #     idxs.append(random.randint(0, len(trainset) - 1))
    print(idxs)
    for it in iters:
        print(f"iter {it}: ", end="", flush=True)
        c_worked = 0
        c_failed = 0
        c_failures = []
        for a in range(0, 1):
            idx = random.randint(0, len(trainset) - 1)
            # source and target for testing
            source_img, source_class = trainset[idxs[a]] # frog
            target_class = random.randint(0, 9)
            while (target_class == source_class):
                target_class = random.randint(0, 9)

            adv_img = part_3(
                source_img,
                target_class,
                ensemble_1,
                ensemble_2,
                ensemble_3,
                device,
            )

            # test output
            img_byte_arr = io.BytesIO()
            adv_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
            response = requests.post("http://floo.cs.uchicago.edu/hw2_black_box", files=files)
            if not get_image_diff(source_img, adv_img):
                print("\033[91mD\033[0m", end="", flush=True)
                c_failed += 1
                c_failures.append((idx, "diff."))
            elif response.json()["class"] == target_class:
                print("\033[92m.\033[0m", end="", flush=True)
                # c_failures.append((target_class, source_class, response.json()["output"][target_class] - response.json()["output"][source_class]))
                c_worked += 1
            else:
                print("\033[91mX\033[0m", end="", flush=True)
                c_failures.append((idxs[a], target_class, source_class, response.json()["output"][target_class] - response.json()["output"][source_class]))
                c_failed += 1
        print("")
        print(c_worked + c_failed, c_worked, c_failed)
        for f in c_failures:
            print(f)

# <<< END TESTING PART 3 <<<

# >>> TESTING BONUS >>>
source_img, source_class = trainset[6] # frog

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

if response.json()["class"] == target_class:
    print("Bonus worked")
else:
    print("Bonus did not work")

# <<< END TESTING BONUS
