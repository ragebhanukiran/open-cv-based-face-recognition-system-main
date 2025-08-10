# Face Recognition Attendance System

A Python-based face recognition system using OpenCV for capturing, training, and tracking faces to record attendance.

## Features
- Capture face images for training
- Train a face recognition model
- Track faces in real-time and log attendance
- GUI built with Tkinter
- Attendance data saved in CSV format

## Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`:
  ```
  opencv-python
  numpy
  pillow
  pandas
  python-dateutil
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ragebhanukiran/open-cv-based-face-recognition-system-main.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `haarcascade_frontalface_default.xml` is in the project directory (download from OpenCV's GitHub if needed).

## Usage
1. Run the application:
   ```bash
   python train.py
   ```
2. GUI Instructions:
   - **Take Images**: Enter ID and Name, then capture up to 60 face images.
   - **Train Images**: Train the model using captured images.
   - **Track Images**: Start real-time face tracking and attendance logging.
   - **Quit**: Exit the application.

## Directory Structure
- `TrainingImage/`: Stores captured face images.
- `StudentDetails/StudentDetails.csv`: Stores ID and Name data.
- `TrainingImageLabel/Trainner.yml`: Trained face recognition model.
- `Attendance/`: Stores attendance CSV files.

## Notes
- Ensure a working webcam is connected.
- Place `haarcascade_frontalface_default.xml` in the root directory.
- The system uses LBPHFaceRecognizer for face recognition.

## License
MIT License. See LICENSE file for details.