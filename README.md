# Face Recognition Attendance System

This repository contains Python scripts for a Face Recognition Attendance System with varying levels of liveness detection.

## Scripts

1.  **`FinalVer.py`**:
    *   Implements a face recognition system for attendance tracking.
    *   Includes basic liveness detection features (texture analysis, eye blinks, micro-movements).
    *   Suitable for general classroom attendance scenarios.

2.  **`embeddings.py`**:
    *   An enhanced version focusing more heavily on robust liveness detection.
    *   Incorporates more advanced checks, including screen detection and session-based analysis (EAR variance).
    *   Designed to be more resilient against presentation attacks (photos, screen replays).

## Setup

1.  **Dependencies**:
    *   Ensure you have Python installed (>= 3.8 recommended).
    *   Install required libraries:
        ```bash
        pip install opencv-python torch numpy Pillow tkinter pandas scikit-learn psutil dlib torchvision facenet-pytorch openpyxl imutils
        ```
    *   If using a GPU for faster processing, ensure you have CUDA and cuDNN installed correctly and install the CUDA-enabled version of PyTorch.

2.  **Facial Landmark Model**:
    *   Both scripts require the dlib facial landmark predictor model for liveness detection.
    *   Download the `shape_predictor_68_face_landmarks.dat` file. You can usually find this file online (search for "dlib 68 face landmarks dat download").
    *   Place the downloaded `.dat` file in the same directory as the Python scripts. The scripts will automatically attempt to load it.

3.  **Known Faces**:
    *   Create a folder named `known_faces` in the same directory as the scripts.
    *   Inside `known_faces`, create sub-folders for each person you want to recognize. The name of each sub-folder should be the person's ID (e.g., student ID).
    *   Place one or more clear images of each person's face (prefered 3) inside their respective ID folder (e.g., `known_faces/10001/face1.jpg`). Supported formats are `.png`, `.jpg`, `.jpeg`.
    *   The system will process these images and create an `embeddings_cache.pkl` file for faster loading on subsequent runs.

4.  **Attendance Sheet**:
    *   Create an Excel file named `Attendence_Sheet.xlsx` in the script directory.
    *   This file should have at least two columns: `ID` and `Name`.
    *   Populate this sheet with the IDs and names of the students/individuals to be tracked. The IDs must match the folder names in `known_faces`.

5.  **Log Folder**:
    *   A `logs_folder` will be created automatically to store temporary files or logs if needed.

## Running the System

1.  Navigate to the directory containing the scripts in your terminal.
2.  Run the desired script using Python:
    *   For standard recognition: `python FinalVer.py`
    *   For enhanced liveness detection: `python embeddings.py`

## Using the GUI

1.  **Camera Feeds**: The main window displays camera feeds. By default, it attempts to use video files specified in the script (`Classroom1.mp4`, etc. in `FinalVer.py` or `FAKE.mp4`, etc. in `embeddings.py`). You can modify the `self.camera_sources` list in the script to use different video files or webcam indices (e.g., `0`, `1`).
2.  **Camera Controls**: Use the checkboxes in the "Controls" section to activate/deactivate specific camera feeds.
3.  **Start/Stop Recognition**:
    *   Click "Start Recognition" to begin detecting and identifying faces.
    *   Click "Stop Recognition" to pause the process and finalize attendance based on the session's data.
4.  **Attendance List**: The left panel shows the list of recognized individuals, their status (Processing, Present, Absent), timestamp, confidence score, and liveness status (`embeddings.py` only).
5.  **Statistics**: Various statistics (Total Students, Present, Absent, Unknown Faces, Fake Faces, Attendance Rate) are displayed below the camera feeds and in the summary section.
6.  **Status Log**: The text box in the bottom-right panel shows real-time status messages, warnings, and errors.
7.  **Export Attendance**: Click "Export Attendance" (after stopping recognition) to save the current session's attendance data to a new Excel file (e.g., `Attendance_YYYY-MM-DD.xlsx`) and a summary text file (`Summary_YYYY-MM-DD.txt`).
8.  **Clear Attendance**: Resets the attendance list and statistics in the GUI.
9.  **Exit**: Closes the application.

## Notes

*   The system accepts both videos and live cameras (or cctv), you just need to follow the instrcution in each file on how to use these sources. 
*   Performance depends heavily on your system hardware (CPU, GPU, RAM) and the number of cameras/faces being processed. The scripts include adaptive frame skipping based on CPU load and processing time.
*   Liveness detection accuracy can vary depending on lighting conditions, camera quality, and the sophistication of spoofing attempts.
