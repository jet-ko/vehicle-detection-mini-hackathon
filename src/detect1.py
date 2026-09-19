from ultralytics import YOLO


model = YOLO("yolo11m.pt")

results = model.predict(
    "input/image.png",
    conf=0.20,
    imgsz=1280,
    save=True,
    name="detect1",
    exist_ok=True,
)

car_count = 0
bus_count = 0
truck_count = 0
motorcycle_count = 0
bicycle_count = 0

for result in results:
    for box in result.boxes:
        class_number = int(box.cls[0])
        vehicle = model.names[class_number]
        kakuritsu = float(box.conf[0])

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
print("Vehicle counts:")
print("car:", car_count)
print("bus:", bus_count)
print("truck:", truck_count)
print("motorcycle:", motorcycle_count)
print("bicycle:", bicycle_count)
print()
print("The result image is saved in runs/detect/detect1.")