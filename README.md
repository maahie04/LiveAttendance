# Live Attendance System using Facial Recognition
This project is a real-time attendance system that uses facial recognition to mark student attendance and records it. The application is built with Streamlit, OpenCV, and the face_recognition library.

## Prerequisites
Before you begin, ensure you have Python installed. You will need to install the following libraries. You can install them using pip:

```bash
pip install face_recognition
pip install streamlit
pip install opencv-python
pip install pandas openpyxl
```

## Setup Instructions
Follow these steps to set up the project directory correctly.

### 1. Student Images Folder
Create a subfolder in the main project directory and name it images.

Inside this images folder, place the photographs of all the students.

Each image file must be named in the following format: Name_RollNo.jpg (e.g., JohnDoe_101.jpg, JaneSmith_102.png).
```text
Project-Folder/
│
├── images/
│   ├── JohnDoe_101.jpg
│   ├── JaneSmith_102.png
│   └── ...
│
└── ...
```
### 2. Student Data File
Create an Excel file (e.g., students.xlsx) in the main project folder.

This file should contain a column with the header regno (or similar).

List the registration numbers of all the students in this column. This will be used to initialize the attendance sheet.

## How to Run the System
Follow these two steps to get the application running.

### Step 1: Generate Face Encodings
First, you need to process the student images to create facial encodings. Run the face_enc.py script from your terminal:
```bash
python face_enc.py
```
This script will read all the images from the images folder, compute the facial encodings for each student, and save them into a file named encodings.txt. This file is crucial for the recognition process.

### Step 2: Launch the Application
Once the encodings.txt file has been generated, you can start the web application. Run the following command in your terminal:
```bash
streamlit run app.py
```
This will launch the Streamlit application in your default web browser, where you can start taking live attendance.

## Preview
Here is a preview of the application interface:

<img width="1919" height="912" alt="Screenshot 2025-09-04 193005" src="https://github.com/user-attachments/assets/3674b631-3881-4180-9648-70a2ee802e89" />

Start Attendance will allow you to start the attendance. A live webcam feed will identify students in real time and mark them present automatically.

Stop and Save Attendance will allow you to download the attendance list and will modify the main excel with the register numbers. 

In case of discrepency ( misidentification of student or unable to identify student) manual override allows student to enter register number to mark absent or present.
