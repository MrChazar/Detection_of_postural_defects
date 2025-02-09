FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

COPY model/model.pkl /app/model/model.pkl
COPY notebooks/yolov5/runs/train/exp7/weights/last.pt /app/notebooks/yolov5/runs/train/exp7/weights/last.pt

CMD ["streamlit", "run", "app.py"]