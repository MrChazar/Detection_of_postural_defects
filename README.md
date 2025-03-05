# Detection_of_postural_defects

This project leverages **YOLOv5** and **neural networks** to detect postural defects, specifically **spondylolisthesis** and **scoliosis**. The goal is to provide an semi professional diagnosis of postural issues using computer vision techniques.

## Features
- **YOLOv5 Integration**: Utilizes the YOLOv5 model for object detection and defect localization.
- **Neural Networks**: Implements deep learning models for classification and analysis of postural defects.
- **User-Friendly Interface**: Built with **Streamlit** for an intuitive and interactive user experience.
- **Docker Support**: Easily deploy and run the application in a containerized environment.

## Screenshots
![image](https://github.com/user-attachments/assets/30ecbae6-e546-45fa-9766-c109873970e0)
![image](https://github.com/user-attachments/assets/6f1e05c8-95ec-4191-8ac1-1527bf847540)



## How to Run the Project

### Prerequisites
- **Docker** installed on your machine. If you don't have Docker, download it from [here](https://www.docker.com/get-started).
- **Git** installed to clone the repository.

### Step 1: Clone the Repository
Clone the project repository to your local machine:
git clone https://github.com/your-username/Detection_of_postural_defects.git
cd Detection_of_postural_defects

### Step 2: Build docker image
docker build -t postural_defects_app .

### Step 3: Run docker container
docker run -p 8501:8501 postural_defects_app
