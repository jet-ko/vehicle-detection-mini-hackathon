import os
import cv2
import timm
import torch

from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


os.makedirs("runs/detect/sedan3", exist_ok=True)

print("Detecting vehicles...")

detector = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11m.pt",
    confidence_threshold=0.20,
    device="cpu",
)

result = get_sliced_prediction(
    image="input/image.png",
    detection_model=detector,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.30,
    overlap_width_ratio=0.30,
    perform_standard_pred=True,
    postprocess_class_agnostic=True,
    postprocess_match_threshold=0.30,
)

print("Loading body type model...")

weightfile = hf_hub_download(
    repo_id="yigechundong/car-body-classifier",
    filename="model.safetensors",
)

bodymodel = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=False,
    num_classes=8,
)

weights = load_file(weightfile)
bodymodel.load_state_dict(weights)
bodymodel.eval()

transform = transforms.Compose(
    [
        transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

labels = [
    "suv",
    "van",
    "station wagon",
    "micro",
    "open wheel",
    "sedan",
    "hatchback",
    "pickup",
]

valid_ids = [0, 1, 2, 3, 5, 6, 7]

vehicle_count = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0,
    "bicycle": 0,
}

type_number = {
    "suv": 0,
    "van": 0,
    "station wagon": 0,
    "micro": 0,
    "sedan": 0,
    "hatchback": 0,
    "pickup": 0,
}

img = cv2.imread("input/image.png")

for prediction in result.object_prediction_list:
    vehicle = prediction.category.name

    if vehicle not in vehicle_count:
        continue

    vehicle_count[vehicle] += 1

    x1, y1, x2, y2 = prediction.bbox.to_xyxy()
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if vehicle == "car":
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        input_img = transform(crop).unsqueeze(0)

        with torch.no_grad():
            output = bodymodel(input_img)
            probabilities = torch.softmax(
                output[0] / 0.565484,
                dim=0,
            )

        ok_probabilities = probabilities[valid_ids]
        ok_probabilities = (
            ok_probabilities / ok_probabilities.sum()
        )

        best = torch.argmax(ok_probabilities).item()
        label_id = valid_ids[best]
        label = labels[label_id]
        kakuritsu = ok_probabilities[best].item()

        type_number[label] += 1

        if label == "sedan":
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        text = label + " " + str(round(kakuritsu, 2))
        print(label, round(kakuritsu, 3))

    else:
        color = (255, 150, 0)
        text = vehicle

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        text,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )

cv2.imwrite("runs/detect/sedan3/result.png", img)

print()
print("Vehicle counts:")

for vehicle in vehicle_count:
    print(vehicle + ":", vehicle_count[vehicle])

print()
print("Car body types:")

for label in type_number:
    if type_number[label] > 0:
        print(label + ":", type_number[label])

print()
print("Sedan cars:", type_number["sedan"])
print("The result image is saved in runs/detect/sedan3")