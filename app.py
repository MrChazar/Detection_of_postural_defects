import streamlit as st
import numpy as np
import cv2
import pickle
import torch
from torch import tensor
import platform
import pathlib
import keras

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# fix na błąd windows path
scoliosis_path = str(pathlib.Path(r"/apka/yolov5/runs/train/exp2/weights/best.pt").resolve())
spondylolisthesis_path = str(pathlib.Path(r"/apka/yolov5/runs/train/exp4/weights/best.pt").resolve())

# Wczytanie modeli
with open('model/model.pkl', 'rb') as f:
    conv_model = pickle.load(f)

# Wczytanie modeli YOLOv5 dla skoliozy i kręgozmyku
yolo_model_scoliosis = torch.hub.load('yolov5', 'custom', path=scoliosis_path, source='local')
yolo_model_spondylolisthesis = torch.hub.load('yolov5','custom', path=spondylolisthesis_path, source='local')


def preprocess_image_conv_model(image):
    image = cv2.resize(image, (120, 60))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def draw_bounding_boxes_scoliosis(image, results):
    for *box, conf, cls in results.xyxy[0]:
        label = f'{results.names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 123, 0), 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 123, 0), 2)
    return image


def draw_bounding_boxes_spondylolisthesis(image, results):
    # Kręgi sortowane od dołu do góry
    sorted_boxes = sorted(results.xyxy[0], key=lambda box: (box[1] + box[3]) / 2)

    # Przetwarzaj kręgi parami
    spondylolisthesis_detected = False
    laterolisthesis_type = None

    for i in range(len(sorted_boxes) - 1):
        print(f"zawartość sorted_boxes {sorted_boxes}")
        # Górny kręg (superior)
        x1_sup, y1_sup, x2_sup, y2_sup = map(int, sorted_boxes[i][:4])
        conf_sup = sorted_boxes[i][4]
        cls_sup = int(sorted_boxes[i][5])
        label_sup = f'{results.names[int(cls_sup)]} {conf_sup:.2f}'

        # Dolny kręg (inferior)
        x1_inf, y1_inf, x2_inf, y2_inf = map(int, sorted_boxes[i + 1][:4])
        conf_inf = sorted_boxes[i + 1][4]
        cls_inf = int(sorted_boxes[i + 1][5])
        label_inf = f'{results.names[int(cls_inf)]} {conf_inf:.2f}'
        print(f"cofy {conf_sup} i {conf_inf}")

        # Obliczenie środka kręgów
        center_sup = ((x1_sup + x2_sup) / 2, (y1_sup + y2_sup) / 2)
        center_inf = ((x1_inf + x2_inf) / 2, (y1_inf + y2_inf) / 2)

        # Obliczenie odległości w osi X
        A = abs(center_sup[0] - center_inf[0])  # Odległość w osi X
        B = abs(x2_inf - x1_inf)  # Szerokość dolnego kręgu

        # Obliczenie stosunku A/B
        ratio = (A / B) * 100  # Procentowe przesunięcie

        # Klasyfikacja Meyerdinga i wykrywanie laterolisthesis
        if ratio > 25:
            spondylolisthesis_detected = True
            if center_sup[0] < center_inf[0]:  # Przesunięcie w lewo
                laterolisthesis_type = "Lewostronny kręgozmyk"
            else:  # Przesunięcie w prawo
                laterolisthesis_type = "Prawostronny kręgozmyk"

        cv2.rectangle(image, (x1_sup, y1_sup), (x2_sup, y2_sup), (0, 255, 0), 3)
        cv2.rectangle(image, (x1_inf, y1_inf), (x2_inf, y2_inf), (0, 255, 0), 3)

        cv2.putText(image, label_sup, (x1_sup, y1_sup - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)
        cv2.putText(image, label_inf, (x1_inf, y1_inf - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)

    return image, spondylolisthesis_detected, laterolisthesis_type


st.title("Model wykrywania wad postawy")

# Panel boczny z ustawieniami
with st.sidebar:
    st.header("Ustawienia")
    uploaded_file = st.file_uploader("Wybierz zdjęcie x-ray:", type=["jpg", "jpeg", "png"])
    tasks_to_solve = st.selectbox("Wybierz chorobę do wykrycia:", ['Skolioza', 'Kręgozmyk'])

    # Dostosowanie wyboru modelu w zależności od wybranej choroby
    if tasks_to_solve == 'Skolioza':
        models_to_use = st.multiselect("Wybierz modele do użycia:", ["Konwolucyjny", "YOLOv5"])
    else:  # Kręgozmyk
        models_to_use = ["YOLOv5"]  # Tylko YOLOv5 jest dostępny dla kręgozmyku
        st.write("Dla kręgozmyku dostępny jest tylko model YOLOv5.")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_with_boxes = image

    col1, col2 = st.columns(2)
    with col1:
        if "Konwolucyjny" in models_to_use and tasks_to_solve == 'Skolioza':
            processed_image_conv = preprocess_image_conv_model(image)
            prediction_conv = conv_model.predict(processed_image_conv)
            st.write("### Wynik konwolucyjny")

            if prediction_conv[0][0] > 0.75:
                st.warning(f"Wykryto Skoliozę! Pewność: {prediction_conv[0][0]:.2f}")
            else:
                st.success(f"Nie wykryto Skoliozy! Pewność: {prediction_conv[0][0]:.2f}")

        if "YOLOv5" in models_to_use:
            if tasks_to_solve == 'Skolioza':
                prediction_yolo = yolo_model_scoliosis(image)
                st.write("### Wyniki YoloV5")
                image_with_boxes = draw_bounding_boxes_scoliosis(image.copy(), prediction_yolo)
                # Wyświetl wyniki detekcji

                if prediction_yolo.xyxy[0].numel() == 0:
                    st.write("Nic nie wykryto")

                for result in prediction_yolo.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = result
                    class_name = prediction_yolo.names[int(cls)]
                    if "scoliosis" in class_name:
                        st.write(f"Wykryto skoliozę! Pewność: {conf:.4f}")
                    else:
                        st.write("Nic nie wykryto")

            elif tasks_to_solve == 'Kręgozmyk':
                prediction_yolo = yolo_model_spondylolisthesis(image)
                st.write("### Wyniki YoloV5")

                image_with_boxes, spondylolisthesis_detected, laterolisthesis_type = draw_bounding_boxes_spondylolisthesis(
                    image.copy(), prediction_yolo)

                # Wyświetl informacje o kręgozmyku
                if spondylolisthesis_detected:
                    st.warning(f"Wykryto kręgozmyk! Typ: {laterolisthesis_type}")
                else:
                    st.success("Nie wykryto kręgozmyku.")

    with col2:
        if models_to_use != []:
            st.image(image_with_boxes, caption='Przesłane zdjęcie', use_container_width=True)