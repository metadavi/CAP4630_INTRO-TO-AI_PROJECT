# CAP4630_INTRO-TO-AI_PROJECT

# AI Vehicle Access Control (ALPR)

An AI-powered access control system that automatically validates vehicle entry using license plate recognition and confidence-based decision logic.

The system detects a vehicle, reads the license plate using OCR, and determines whether access should be granted based on a whitelist and reliability threshold. Multi-frame analysis is used to improve accuracy under motion blur, lighting changes, and occlusion.

---

## Motivation
Traditional gate systems rely on RFID cards or manual guards. These systems are vulnerable to:
- Lost or shared credentials
- Human error
- Plate spoofing
- Poor visibility conditions

This project explores how computer vision can create a reliable, automated, and auditable vehicle authentication system.

---

## Features
- License plate detection using object detection
- Optical Character Recognition (OCR) for plate reading
- Multi-frame voting for accuracy improvement
- Confidence-based decision making
- Whitelist validation
- Event logging (allowed / denied / uncertain)

---

## System Pipeline
1. Capture video frame(s)
2. Detect vehicle and license plate
3. Crop plate region
4. Run OCR to extract characters
5. Aggregate predictions across frames
6. Compare with authorized database
7. Output decision

---

## Dataset Sources
- Kaggle
- UCI Machine Learning Repository
- Roboflow Universe
- Hugging Face Datasets

(Custom labeled samples may also be used for evaluation)

---

## Tech Stack
- Python
- OpenCV
- PyTorch
- YOLO (object detection)
- OCR model (Tesseract / CRNN / EasyOCR)
- NumPy / Pandas

---

## Project Structure
