import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import logging
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition Attendance System")
        self.window.geometry('1280x720')
        self.window.configure(background='#2c3e50')

        # Initialize camera
        self.cam = None
        self.preview_active = False
        self.frame_queue = queue.Queue(maxsize=1)  # Queue for frame processing

        # Create main frame
        self.main_frame = tk.Frame(self.window, bg='#2c3e50')
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="Face Recognition Attendance System",
            bg="#27ae60", 
            fg="white", 
            font=('Helvetica', 24, 'bold'),
            pady=10
        )
        self.title_label.pack(fill='x')

        # Buttons frame
        self.btn_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        self.btn_frame.pack(pady=10)
        self.btn_take = tk.Button(self.btn_frame, text="Take Images", command=self.take_images, fg="white", bg="#27ae60", font=('Helvetica', 14, 'bold'), width=15)
        self.btn_take.grid(row=0, column=0, padx=10)
        self.btn_train = tk.Button(self.btn_frame, text="Train Images", command=self.train_images, fg="white", bg="#27ae60", font=('Helvetica', 14, 'bold'), width=15)
        self.btn_train.grid(row=0, column=1, padx=10)
        self.btn_track = tk.Button(self.btn_frame, text="Track Images", command=self.track_images, fg="white", bg="#27ae60", font=('Helvetica', 14, 'bold'), width=15)
        self.btn_track.grid(row=0, column=2, padx=10)
        self.btn_quit = tk.Button(self.btn_frame, text="Quit", command=self.quit_app, fg="white", bg="#e74c3c", font=('Helvetica', 14, 'bold'), width=15)
        self.btn_quit.grid(row=0, column=3, padx=10)

        # Camera preview canvas
        self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg='black')
        self.canvas.pack(pady=20)

        # Input frame
        self.input_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        self.input_frame.pack(fill='x', pady=10)

        # ID input
        self.lbl_id = tk.Label(self.input_frame, text="Enter ID", width=15, fg="white", bg="#3498db", font=('Helvetica', 14, 'bold'))
        self.lbl_id.grid(row=0, column=0, padx=5, pady=5)
        self.txt_id = tk.Entry(self.input_frame, width=20, font=('Helvetica', 14))
        self.txt_id.grid(row=0, column=1, padx=5, pady=5)
        self.btn_clear_id = tk.Button(self.input_frame, text="Clear", command=self.clear_id, fg="white", bg="#e74c3c", font=('Helvetica', 12))
        self.btn_clear_id.grid(row=0, column=2, padx=5)

        # Name input
        self.lbl_name = tk.Label(self.input_frame, text="Enter Name", width=15, fg="white", bg="#3498db", font=('Helvetica', 14, 'bold'))
        self.lbl_name.grid(row=1, column=0, padx=5, pady=5)
        self.txt_name = tk.Entry(self.input_frame, width=20, font=('Helvetica', 14))
        self.txt_name.grid(row=1, column=1, padx=5, pady=5)
        self.btn_clear_name = tk.Button(self.input_frame, text="Clear", command=self.clear_name, fg="white", bg="#e74c3c", font=('Helvetica', 12))
        self.btn_clear_name.grid(row=1, column=2, padx=5)

        # Notification
        self.lbl_notification = tk.Label(self.main_frame, text="Notification: ", fg="white", bg="#3498db", font=('Helvetica', 14, 'bold'))
        self.lbl_notification.pack(pady=5)
        self.notification = tk.Label(self.main_frame, text="", fg="white", bg="#2c3e50", font=('Helvetica', 12), wraplength=800)
        self.notification.pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.main_frame, maximum=60, length=300)
        self.progress.pack(pady=5)

        # Attendance display
        self.lbl_attendance = tk.Label(self.main_frame, text="Attendance: ", fg="white", bg="#3498db", font=('Helvetica', 14, 'bold'))
        self.lbl_attendance.pack(pady=5)
        self.attendance_text = tk.Text(self.main_frame, height=5, width=60, font=('Helvetica', 12), bg="#ecf0f1", fg="black")
        self.attendance_text.pack()

        # Start camera preview
        self.start_camera_preview()

    def clear_id(self):
        self.txt_id.delete(0, 'end')
        self.notification.config(text="")

    def clear_name(self):
        self.txt_name.delete(0, 'end')
        self.notification.config(text="")

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def start_camera_preview(self):
        if not self.preview_active:
            self.cam = cv2.VideoCapture(0)
            if not self.cam.isOpened():
                self.notification.config(text="Error: Could not open camera")
                logging.error("Failed to open camera")
                return
            self.preview_active = True
            self.update_camera_preview()

    def update_camera_preview(self):
        if self.preview_active:
            if not self.frame_queue.full():
                ret, frame = self.cam.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 480))
                    self.frame_queue.put(frame)
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(30, self.update_camera_preview)  # Reduced update frequency

    def stop_camera_preview(self):
        if self.preview_active:
            self.preview_active = False
            if self.cam:
                self.cam.release()
                self.cam = None
            self.canvas.delete("all")
            while not self.frame_queue.empty():
                self.frame_queue.get()  # Clear queue

    def take_images(self):
        def capture():
            Id = self.txt_id.get().strip()
            name = self.txt_name.get().strip()
            if not self.is_number(Id):
                self.notification.config(text="Enter Numeric ID")
                return
            if not name.isalpha():
                self.notification.config(text="Enter Alphabetical Name")
                return

            self.stop_camera_preview()
            self.cam = cv2.VideoCapture(0)
            if not self.cam.isOpened():
                self.notification.config(text="Error: Could not open camera")
                logging.error("Failed to open camera")
                return

            harcascadePath = "haarcascade_frontalface_default.xml"
            if not os.path.exists(harcascadePath):
                self.notification.config(text="Error: Haar cascade file not found")
                logging.error("Haar cascade file not found")
                return
            detector = cv2.CascadeClassifier(harcascadePath)
            if detector.empty():
                self.notification.config(text="Error: Could not load Haar cascade")
                logging.error("Failed to load Haar cascade")
                return

            sampleNum = 0
            os.makedirs("TrainingImage", exist_ok=True)
            self.progress['value'] = 0
            self.window.update()
            while True:
                ret, img = self.cam.read()
                if not ret:
                    self.notification.config(text="Error: Failed to capture image")
                    logging.error("Failed to capture image")
                    break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))  # Optimized parameters
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    sampleNum += 1
                    if sampleNum <= 60:  # Limit to 60 images
                        filename = f"TrainingImage/{name}.{Id}.{sampleNum}.jpg"
                        cv2.imwrite(filename, gray[y:y+h, x:x+w])
                        logging.info(f"Saved image: {filename}")
                if len(faces) == 0:
                    self.notification.config(text="No faces detected, please adjust position")
                cv2.imshow('Capturing', img)
                self.progress['value'] = sampleNum
                self.window.update()
                if cv2.waitKey(150) & 0xFF == ord('q') or sampleNum >= 60:  # Increased delay
                    break
            self.cam.release()
            cv2.destroyAllWindows()

            os.makedirs("StudentDetails", exist_ok=True)
            csv_path = 'StudentDetails/StudentDetails.csv'
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(['Id', 'Name'])
            with open(csv_path, 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([Id, name])
            self.notification.config(text=f"Images Saved for ID: {Id}, Name: {name} ({sampleNum} images)")
            logging.info(f"Images saved for ID: {Id}, Name: {name}, Count: {sampleNum}")
            self.start_camera_preview()

        threading.Thread(target=capture, daemon=True).start()

    def train_images(self):
        def train():
            self.stop_camera_preview()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            harcascadePath = "haarcascade_frontalface_default.xml"
            if not os.path.exists(harcascadePath):
                self.notification.config(text="Error: Haar cascade file not found")
                logging.error("Haar cascade file not found")
                return
            detector = cv2.CascadeClassifier(harcascadePath)
            if detector.empty():
                self.notification.config(text="Error: Could not load Haar cascade")
                logging.error("Failed to load Haar cascade")
                return

            faces, ids = self.get_images_and_labels("TrainingImage")
            print(f"Loaded {len(faces)} faces for training")
            if len(faces) < 1:
                self.notification.config(text="Error: No valid training images found")
                logging.error("No valid training images found")
                return

            try:
                # Batch training for larger datasets
                batch_size = 20
                for i in range(0, len(faces), batch_size):
                    batch_faces = faces[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]
                    recognizer.update(batch_faces, np.array(batch_ids))
                os.makedirs("TrainingImageLabel", exist_ok=True)
                recognizer.save("TrainingImageLabel/Trainner.yml")
                print(f"Model saved to TrainingImageLabel/Trainner.yml")
                self.notification.config(text=f"Trained {len(faces)} images successfully")
                logging.info(f"Trained {len(faces)} images successfully")
            except Exception as e:
                self.notification.config(text=f"Error during training: {str(e)}")
                logging.error(f"Training error: {str(e)}")
            self.start_camera_preview()

        threading.Thread(target=train, daemon=True).start()

    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        ids = []
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                filename = os.path.basename(imagePath)
                Id = int(filename.split(".")[1])  # Assumes format: name.ID.sampleNum.jpg
                faces.append(imageNp)
                ids.append(Id)
            except Exception as e:
                logging.error(f"Error processing image {imagePath}: {str(e)}")
        return faces, ids

    def track_images(self):
        def track():
            self.stop_camera_preview()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read("TrainingImageLabel/Trainner.yml")
            except Exception as e:
                self.notification.config(text="Error: Trained model not found")
                logging.error(f"Error loading model: {str(e)}")
                self.start_camera_preview()
                return

            harcascadePath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(harcascadePath)
            if faceCascade.empty():
                self.notification.config(text="Error: Could not load Haar cascade")
                logging.error("Failed to load Haar cascade")
                self.start_camera_preview()
                return

            try:
                df = pd.read_csv("StudentDetails/StudentDetails.csv")
                if 'Id' not in df.columns or 'Name' not in df.columns:
                    self.notification.config(text="Error: Invalid CSV format, missing 'Id' or 'Name' columns")
                    logging.error("Invalid CSV format")
                    self.start_camera_preview()
                    return
            except Exception as e:
                self.notification.config(text="Error: Could not load student details")
                logging.error(f"Error loading student details: {str(e)}")
                self.start_camera_preview()
                return

            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                self.notification.config(text="Error: Could not open camera")
                logging.error("Failed to open camera")
                self.start_camera_preview()
                return

            col_names = ['Id', 'Name', 'Date', 'Time']
            attendance = pd.DataFrame(columns=col_names)
            print("Starting face tracking...")

            while True:
                ret, im = cam.read()
                if not ret:
                    self.notification.config(text="Error: Failed to capture image")
                    logging.error("Failed to capture image")
                    break

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    print(f"Detected Id: {Id} with confidence: {conf}")

                    if conf < 50:
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        try:
                            aa = df.loc[df['Id'] == Id]['Name'].iloc[0]
                            tt = f"{Id}-{aa}"
                            # Add attendance only once per session per ID
                            if Id not in attendance['Id'].values:
                                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                                print(f"Attendance recorded for {aa} (ID: {Id}) at {date} {timeStamp}")
                        except IndexError:
                            tt = "Unknown"
                    else:
                        tt = "Unknown"

                    cv2.putText(im, str(tt), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Tracking', im)
                self.attendance_text.delete(1.0, tk.END)
                self.attendance_text.insert(tk.END, attendance.to_string(index=False))

                if cv2.waitKey(30) == ord('q'):
                    print("Stopping tracking loop")
                    break

            if not attendance.empty:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour, Minute, Second = timeStamp.split(":")
                os.makedirs("Attendance", exist_ok=True)
                fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
                attendance.to_csv(fileName, index=False)
                print(f"Attendance saved to {fileName}")
                self.notification.config(text=f"Attendance saved to {fileName}")
                logging.info(f"Attendance saved to {fileName}")
            else:
                print("No attendance to save.")
                self.notification.config(text="No attendance recorded")
                logging.info("No attendance recorded")

            cam.release()
            cv2.destroyAllWindows()
            self.start_camera_preview()

        threading.Thread(target=track, daemon=True).start()




    def quit_app(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.stop_camera_preview()
            self.window.destroy()

if __name__ == "__main__":
    window = tk.Tk()
    app = FaceRecognitionApp(window)
    window.mainloop()
