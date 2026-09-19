import os

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


os.makedirs("runs/detect/sahi2", exist_ok=True)

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11m.pt",
    confidence_threshold=0.20,
    device="cpu",
)

result = get_sliced_prediction(
    image="input/image.png",
    detection_model=model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.30,
    overlap_width_ratio=0.30,
    perform_standard_pred=True,
    postprocess_class_agnostic=True,
    postprocess_match_threshold=0.30,
)

result.export_visuals(
    export_dir="runs/detect/sahi2",
    file_name="image",
)

car_count = 0
bus_count = 0
truck_count = 0
motorcycle_count = 0
bicycle_count = 0

for prediction in result.object_prediction_list:
    vehicle = prediction.category.name
    kakuritsu = prediction.score.value

    if vehicle == "car":
        car_count += 1
    elif vehicle == "bus":
        bus_count += 1
    elif vehicle == "truck":
        truck_count += 1
    elif vehicle == "motorcycle":
        motorcycle_count += 1
    elif vehicle == "bicycle":
        bicycle_count += 1
    else:
        continue

    print(vehicle, round(kakuritsu, 3))

print()
print("Vehicle counts with SAHI:")
print("car:", car_count)
print("bus:", bus_count)
print("truck:", truck_count)
print("motorcycle:", motorcycle_count)
print("bicycle:", bicycle_count)
print()
print("The result image is saved in runs/detect/sahi2.")