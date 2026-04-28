# Drone-Based Early Forest Fire Detection

## Overview
This project focuses on developing a drone-based system for early forest fire detection using computer vision and machine learning. By leveraging RGB and thermal imagery, the system aims to identify early signs of fire such as smoke and heat signatures before fires spread.

The goal is to build a reliable and scalable solution that can assist in wildfire prevention and rapid response.

---

## Objectives
- Detect early signs of fire (smoke, heat) from aerial imagery  
- Compare traditional machine learning vs deep learning approaches  
- Implement real-time detection capabilities for drone deployment  
- Evaluate model performance using standard metrics  

---

## Problem Type
This project evolves through multiple stages:
- Binary Classification → Fire vs No Fire  
- Object Detection → Locate fire and smoke (e.g., YOLO)  
- Optional: Segmentation → Pixel-level fire mapping  

---

## Tech Stack
- Python  
- PyTorch (deep learning)  
- OpenCV (image processing)  
- scikit-learn (baseline models and evaluation)  
- YOLO (Ultralytics) for object detection  
- Optional: ROS / MAVSDK for drone integration  

---

## Dataset
This project uses publicly available wildfire datasets, including:
- UAV-based RGB and thermal fire datasets  
- Smoke detection image datasets  
- Satellite wildfire datasets (optional extension)  

Datasets include labeled images or videos for fire, smoke, and non-fire scenarios.

---

## Methodology
1. Data Preprocessing  
   - Clean and normalize images  
   - Split into training, validation, and test sets  

2. Baseline Model (scikit-learn)  
   - Train simple classifiers such as Random Forest or SVM  

3. Deep Learning Model  
   - Train CNN or YOLO model for detection  
   - Use annotated datasets for supervised learning  

4. Evaluation  
   - Accuracy  
   - Precision and Recall  
   - F1-score  
   - Detection speed (frames per second)  

5. Optional Deployment  
   - Run model on an edge device (e.g., Jetson Nano)  
   - Integrate with drone camera feed  

---

## Expected Results
- Accurate detection of fire and smoke in aerial imagery  
- Faster detection compared to traditional methods  
- Reduced false positives through model tuning  

---

## Future Work
- Real-time drone deployment  
- Autonomous drone patrol system  
- Fire spread prediction models  
- Multi-drone coordination  

---

## Limitations
- Dependent on dataset quality and diversity  
- Environmental noise such as fog, clouds, or lighting conditions may affect accuracy  
- Limited real-world testing due to safety constraints  

---

## Author
Student Researcher – Drone and AI Systems  

---

## License
This project is for academic and research purposes only.