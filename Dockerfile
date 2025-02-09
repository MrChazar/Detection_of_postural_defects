# Użyj oficjalnego obrazu Python jako bazowego
FROM python:3.10-slim

# Ustaw katalog roboczy
WORKDIR /apka

# Skopiuj wymagane pliki do kontenera
COPY . /apka


# Zainstaluj wymagane pakiety
RUN cat requirements.txt
RUN apt update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj modele do kontenera
COPY model/model.pkl /apka/model/model.pkl
COPY yolov5/runs/train/exp2/weights/best.pt /apka/yolov5/runs/train/exp2/weights/best.pt
COPY yolov5/runs/train/exp4/weights/best.pt /apka/yolov5/runs/train/exp4/weights/best.pt

# Uruchom aplikację Streamlit
CMD ["streamlit", "run", "app.py"]