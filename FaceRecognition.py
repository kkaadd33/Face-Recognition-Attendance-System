import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue
import warnings
from torchvision import transforms
import sqlite3
import time
import psutil  
from torch.cuda.amp import autocast  
import dlib  
from scipy.spatial import distance  
import pickle
from concurrent.futures import ThreadPoolExecutor

# Limit OpenCV threads to avoid resource contention with our own threading
try:
    cv2.setNumThreads(1)
    print(f"OpenCV internal threads limited to: {cv2.getNumThreads()}")
except Exception as e:
    print(f"Could not set OpenCV threads (might be fine): {e}")

warnings.filterwarnings('ignore', category=FutureWarning)

class FaceRecognitionApp:
    def __init__(self):
        # Core control flags
        self.recognition_active = False

        # Face tracking infrastructure
        self.active_tracking = set()  # Track which IDs we're monitoring
        self.face_confidence = {}
        self.confidence_history = {}
        
        # Statistics tracking
        self.fake_faces_count = 0
        self.fake_faces_tracking = {} # Faces that failed liveness check
        self.unknown_faces_count = 0
        self.unknown_faces_tracking = {}  # Unique unknown faces

        # Face persistence across frames
        self.face_tracking = {}  # Track faces across frames
        self.tracking_ttl = 30   # Frames to maintain tracking
        self.last_positions = {} # Last known positions
        
        # Performance optimization parameters
        self.cpu_threshold = 90.0  # Adjust skip rate if CPU exceeds this threshold
        self.max_cpu_skip_rate_increase = 5 # Maximum skip rate increase under high CPU load
        
        # Thread management for parallel processing
        self.processing_executor = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() * 3) // 4 if os.cpu_count() else 2))

        # Liveness detection timing
        self.last_liveness_check = {}
        self.liveness_check_interval = 1.0 # Seconds between checks for each tracked face

        # Liveness detection configuration
        self.liveness_history = {}
        self.liveness_threshold = 0.5  # Threshold for considering a face "live"
        self.blink_threshold = 0.2  # Threshold for detecting eye blinks
        self.texture_threshold = 0.7  # Threshold for texture analysis
        self.liveness_ttl = 60  # Frames to maintain liveness status
        
        # Set up facial landmark detection for liveness checks
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            landmark_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(landmark_path):
                self.landmark_predictor = dlib.shape_predictor(landmark_path)
                self.liveness_detection_enabled = True
                print("Liveness detection enabled")
            else:
                print(f"Landmark predictor file not found at {landmark_path}")
                print("Downloading landmark predictor...")
                self.liveness_detection_enabled = False
        except Exception as e:
            print(f"Error initializing liveness detection: {str(e)}")
            self.liveness_detection_enabled = False
        
        # Display and synchronization
        self.display_frames = {}     # Frames being displayed
        self.display_annotations = {} # Annotations for each frame
        self.processing_lock = threading.Lock()  # Protects shared data like annotations
        
        # Confidence calculation parameters
        self.frame_counter = 0
        self.frames_interval = 12  # Calculate confidence every 12 frames
        self.confidence_window = 12  # Window size for confidence calculation
        self.min_confidence_frames = 6  # Minimum frames needed for confidence
        self.confidence_threshold = 60  # Threshold for Present/Absent
        self.max_confidence = 85  # Maximum confidence value

        # Face recognition settings
        self.default_recognition_threshold = 0.5  # Lower threshold catches more potential matches

        # Camera configuration (change the video files paths to your own, also if you want to use live cameras, change these paths to the camera index)
        self.camera_sources = [
            r"C:\\Users\\alman\\OneDrive\\Desktop\\KU\\SDPI\\videos\\Classroom1.mp4",
            r"C:\\Users\\alman\\OneDrive\\Desktop\\KU\\SDPI\\videos\\Classroom2.mp4",
            r"C:\\Users\\alman\\OneDrive\\Desktop\\KU\\SDPI\\videos\\Classroom3.mp4"
        ]
        self.active_cameras = [True, True, True]  # All cameras active by default
        self.cameras = []
        self.camera_frames = {}  # Store frames from each camera
        self.stop_threads = False  # Flag to stop camera threads

        # Hardware acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MTCNN face detection with tuned parameters for speed/accuracy balance
        self.mtcnn = MTCNN(
            image_size=160,
            margin=10,
            min_face_size=40,  # Larger size reduces false positives and improves speed
            thresholds=[0.7, 0.8, 0.8],  # Higher thresholds for stricter detection
            factor=0.709,
            post_process=True,
            keep_all=True, # We'll filter based on probability later
            device=self.device,
            select_largest=False
        )

        # Load face recognition model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Performance optimization settings
        self.detection_skip_rate = 3  # Process every nth frame
        self.adaptive_skip_rate = True # Dynamically adjust based on system load
        self.max_skip_rate = 8  # Maximum frames to skip
        self.min_skip_rate = 2  # Minimum frames to skip
        self.last_processing_time = 0
        self.face_count_history = [0] * 10  # Tracks recent face counts
        
        # Processing limits to maintain performance
        self.max_faces_per_frame = 10 # Maximum faces to process in a single frame
        
        # Embedding cache for faster recognition
        self.embeddings_cache = {}
        self.cache_max_size = 300
        self.cache_ttl = 10
        self.cache_timestamps = {}

        # Thread-safe logging
        self.log_queue = Queue()

        # File paths (chnage to your own paths)
        self.known_faces_folder = r"C:\\Users\\alman\\OneDrive\\Desktop\\venv\\venv\\known_faces"
        self.attendance_file = r"C:\\Users\\alman\\OneDrive\\Desktop\\venv\\venv\\Attendence_Sheet.xlsx"
        self.log_folder = r"C:\\Users\\alman\\OneDrive\\Desktop\\venv\\venv\\logs_folder"

        # Create required folders
        os.makedirs(self.known_faces_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        self.embeddings_cache_file = os.path.join(self.known_faces_folder, 'embeddings_cache.pkl')
        
        # Set up UI
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')  # Light gray background

        # Build UI components
        self.setup_gui()

        # Load reference data
        self.load_student_data()
        self.load_known_faces()

        # Initialize attendance tracking
        self.attendance = {}

        # Start camera feeds (done last to ensure everything is ready)
        self.start_camera()

        # Initialize frame tracking
        self.frame_counter = 0
        self.frames_interval = 12
        self.face_detections = {}  # Track face detections in current window

    def setup_gui(self):
        # Create main frames with default style
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.middle_frame = ttk.Frame(self.root)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # Left Frame ( Attendance Display)
        attendance_frame = ttk.LabelFrame(self.left_frame, text="Today's Attendance")
        attendance_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Attendance Treeview
        self.attendance_tree = ttk.Treeview(
            attendance_frame, 
            columns=('ID', 'Name', 'Status', 'Time', 'Confidence'), 
            show='headings'
        )
        # Set up headings
        self.attendance_tree.heading('ID', text='ID')
        self.attendance_tree.heading('Name', text='Name')
        self.attendance_tree.heading('Status', text='Status')
        self.attendance_tree.heading('Time', text='Time')
        self.attendance_tree.heading('Confidence', text='Confidence')

        # Set column widths
        self.attendance_tree.column('ID', width=100)
        self.attendance_tree.column('Name', width=150)
        self.attendance_tree.column('Status', width=80)
        self.attendance_tree.column('Time', width=100)
        self.attendance_tree.column('Confidence', width=100)

        # Add scrollbar to attendance tree
        attendance_scroll = ttk.Scrollbar(attendance_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=attendance_scroll.set)

        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        attendance_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add summary section under attendance list
        self.summary_frame = ttk.LabelFrame(self.left_frame, text="Session Summary")
        self.summary_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Create summary content with attendance stats and charts
        summary_content = ttk.Frame(self.summary_frame)
        summary_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create two columns for stats
        stats_left = ttk.Frame(summary_content)
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stats_right = ttk.Frame(summary_content)
        stats_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left column - Student stats
        ttk.Label(stats_left, text="Student Statistics", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Add student stats with colored indicators
        student_stats = ttk.Frame(stats_left)
        student_stats.pack(fill=tk.X)
        
        # Total row
        total_row = ttk.Frame(student_stats)
        total_row.pack(fill=tk.X, pady=2)
        ttk.Label(total_row, text="📋 Total:").pack(side=tk.LEFT)
        self.summary_total_var = tk.StringVar(value="0")
        ttk.Label(total_row, textvariable=self.summary_total_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Present row
        present_row = ttk.Frame(student_stats)
        present_row.pack(fill=tk.X, pady=2)
        ttk.Label(present_row, text="✓ Present:").pack(side=tk.LEFT)
        self.summary_present_var = tk.StringVar(value="0")
        ttk.Label(present_row, textvariable=self.summary_present_var, font=("Arial", 9, "bold"), foreground="green").pack(side=tk.LEFT, padx=5)
        
        # Absent row
        absent_row = ttk.Frame(student_stats)
        absent_row.pack(fill=tk.X, pady=2)
        ttk.Label(absent_row, text="✗ Absent:").pack(side=tk.LEFT)
        self.summary_absent_var = tk.StringVar(value="0")
        ttk.Label(absent_row, textvariable=self.summary_absent_var, font=("Arial", 9, "bold"), foreground="red").pack(side=tk.LEFT, padx=5)
        
        # Rate row
        rate_row = ttk.Frame(student_stats)
        rate_row.pack(fill=tk.X, pady=2)
        ttk.Label(rate_row, text="📊 Rate:").pack(side=tk.LEFT)
        self.summary_rate_var = tk.StringVar(value="0%")
        ttk.Label(rate_row, textvariable=self.summary_rate_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Right column - Face detection stats
        ttk.Label(stats_right, text="Face Detection", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        face_stats = ttk.Frame(stats_right)
        face_stats.pack(fill=tk.X)
        
        # Unknown faces row (discard this for now)
        unknown_row = ttk.Frame(face_stats)
        unknown_row.pack(fill=tk.X, pady=2)
        #ttk.Label(unknown_row, text="❓ Unknown:").pack(side=tk.LEFT)
        self.summary_unknown_var = tk.StringVar(value="0")
        #ttk.Label(unknown_row, textvariable=self.summary_unknown_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Fake faces row
        fake_row = ttk.Frame(face_stats)
        fake_row.pack(fill=tk.X, pady=2)
        ttk.Label(fake_row, text="⚠ Fake:").pack(side=tk.LEFT)
        self.summary_fake_var = tk.StringVar(value="0")
        ttk.Label(fake_row, textvariable=self.summary_fake_var, font=("Arial", 9, "bold"), foreground="orange").pack(side=tk.LEFT, padx=5)
        
        # Total detected
        detected_row = ttk.Frame(face_stats)
        detected_row.pack(fill=tk.X, pady=2)
        ttk.Label(detected_row, text="👥 Total Detected:").pack(side=tk.LEFT)
        self.summary_detected_var = tk.StringVar(value="0")
        ttk.Label(detected_row, textvariable=self.summary_detected_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Bar chart showing attendance rate
        self.rate_canvas = tk.Canvas(self.summary_frame, height=20, bg='#f0f0f0', highlightthickness=0)
        self.rate_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Middle Frame - Video Display
        camera_container = ttk.Frame(self.middle_frame)
        camera_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Main camera (large display)
        self.video_label = ttk.Label(camera_container)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label.configure(background='#f0f0f0')

        # Frame for two smaller camera displays
        small_cameras_frame = ttk.Frame(camera_container)
        small_cameras_frame.pack(fill=tk.X, padx=5, pady=5)

        self.video_label2 = ttk.Label(small_cameras_frame)
        self.video_label2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.video_label2.configure(background='#f0f0f0')

        self.video_label3 = ttk.Label(small_cameras_frame)
        self.video_label3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.video_label3.configure(background='#f0f0f0')

        self.video_labels = [self.video_label, self.video_label2, self.video_label3]

        # Add statistics frame to show student counts
        self.stats_frame = ttk.LabelFrame(camera_container, text="Class Statistics")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_grid = ttk.Frame(self.stats_frame)
        stats_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a better layout with status icons
        # Total students
        ttk.Label(stats_grid, text="📋 Total Students:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.total_students_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_students_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Present students
        ttk.Label(stats_grid, text="✓ Present:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.present_students_var = tk.StringVar(value="0")
        present_label = ttk.Label(stats_grid, textvariable=self.present_students_var, font=("Arial", 10, "bold"))
        present_label.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        # Absent students
        ttk.Label(stats_grid, text="✗ Absent:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.absent_students_var = tk.StringVar(value="0")
        absent_label = ttk.Label(stats_grid, textvariable=self.absent_students_var, font=("Arial", 10, "bold"))
        absent_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Unknown faces
        ttk.Label(stats_grid, text="❓ Unknown Faces:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.unknown_faces_var = tk.StringVar(value="0")
        unknown_label = ttk.Label(stats_grid, textvariable=self.unknown_faces_var, font=("Arial", 10, "bold"))
        unknown_label.grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        # Add fake detection counts
        ttk.Label(stats_grid, text="⚠ Fake Faces:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.fake_faces_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.fake_faces_var, font=("Arial", 10, "bold")).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Add attendance rate percentage
        ttk.Label(stats_grid, text="📊 Attendance Rate:").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        self.attendance_rate_var = tk.StringVar(value="0%")
        ttk.Label(stats_grid, textvariable=self.attendance_rate_var, font=("Arial", 10, "bold")).grid(row=2, column=3, sticky="w", padx=5, pady=2)

        # Right Frame (  Controls and Status)
        controls_frame = ttk.LabelFrame(self.right_frame, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(controls_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(padx=5, pady=2)

        self.stop_button = ttk.Button(controls_frame, text="Stop Recognition", command=self.stop_recognition, state='disabled')
        self.stop_button.pack(padx=5, pady=2)

        camera_controls_frame = ttk.LabelFrame(controls_frame, text="Camera Controls")
        camera_controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.camera_vars = []
        for i in range(3):
            var = tk.BooleanVar(value=True)
            self.camera_vars.append(var)
            ttk.Checkbutton(camera_controls_frame, text=f"Camera {i+1}", variable=var, command=self.update_camera_status).pack(anchor=tk.W)

        ttk.Label(controls_frame, text="Recognition Threshold:").pack(padx=5, pady=2)
        self.threshold_var = tk.DoubleVar(value=self.default_recognition_threshold)
        threshold_slider = ttk.Scale(controls_frame, from_=0.3, to=0.8, orient=tk.HORIZONTAL, variable=self.threshold_var)
        threshold_slider.pack(fill=tk.X, padx=5, pady=2)

        self.status_frame = ttk.LabelFrame(self.right_frame, text="Status")
        self.status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text = tk.Text(self.status_frame, height=20, width=40)
        self.status_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(self.right_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(button_frame, text="Export Attendance", command=self.export_attendance).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Attendance", command=self.clear_attendance).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_app).pack(side=tk.RIGHT, padx=5)

        for frame in [self.left_frame, self.middle_frame, self.right_frame]:
            frame.configure(style='TFrame')
        attendance_frame.configure(style='TLabelframe')
        controls_frame.configure(style='TLabelframe')
        self.status_frame.configure(style='TLabelframe')

    def load_known_faces(self):
        import pickle

        # Load or initialize cache
        try:
            if os.path.exists(self.embeddings_cache_file):
                with open(self.embeddings_cache_file, 'rb') as f:
                    file_metadata = pickle.load(f)
                self.log_status("Loaded embeddings cache.")
            else:
                file_metadata = {}
                self.log_status("Embeddings cache not found, building cache.")
        except Exception:
            file_metadata = {}
            self.log_status("Failed to load cache, rebuilding.")

        updated = False
        current_paths = []

        # Scan known faces folder
        for person_id in os.listdir(self.known_faces_folder):
            person_folder = os.path.join(self.known_faces_folder, person_id)
            if not os.path.isdir(person_folder):
                continue

            for image_file in os.listdir(person_folder):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(person_folder, image_file)
                current_paths.append(image_path)
                mod_time = os.path.getmtime(image_path)
                meta = file_metadata.get(image_path)
                if not meta or meta['mod_time'] != mod_time:
                    self.log_status(f"Processing {image_path}...")
                    embeddings = self.process_image(image_path)
                    img = cv2.imread(image_path)

                    bright_embeddings = []
                    if img is not None:
                        bright_img = np.clip(img * 1.2, 0, 255).astype(np.uint8)
                        bright_path = os.path.join(self.log_folder, "temp_bright.jpg")
                        cv2.imwrite(bright_path, bright_img)
                        bright_embeddings = self.process_image(bright_path)
                        if os.path.exists(bright_path):
                            os.remove(bright_path)

                    dark_embeddings = []
                    if img is not None:
                        dark_img = np.clip(img * 0.8, 0, 255).astype(np.uint8)
                        dark_path = os.path.join(self.log_folder, "temp_dark.jpg")
                        cv2.imwrite(dark_path, dark_img)
                        dark_embeddings = self.process_image(dark_path)
                        if os.path.exists(dark_path):
                            os.remove(dark_path)

                    all_embeddings = embeddings + bright_embeddings + dark_embeddings
                    file_metadata[image_path] = {
                        'person_id': person_id,
                        'mod_time': mod_time,
                        'embeddings': all_embeddings
                    }
                    updated = True

        # Remove deleted images from cache
        for path in list(file_metadata):
            if path not in current_paths:
                self.log_status(f"Removing cached embeddings for removed image {path}")
                del file_metadata[path]
                updated = True

        # Rebuild known_faces dict
        self.known_faces = {}
        for meta in file_metadata.values():
            pid = meta['person_id']
            self.known_faces.setdefault(pid, []).extend(meta['embeddings'])

        # Save cache if updated
        if updated:
            try:
                with open(self.embeddings_cache_file, 'wb') as f:
                    pickle.dump(file_metadata, f)
                self.log_status("Updated embeddings cache.")
            except Exception as e:
                self.log_status(f"Failed to save cache: {e}")
        else:
            self.log_status("Embeddings cache is up-to-date.")

        self.log_status("Finished loading known faces.")

    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Advanced image preprocessing for better feature extraction
            # 1) Convert to LAB color space for better color normalization
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            
            # 2) Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
            
            # 3) Convert back to RGB
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 4) Apply slight Gaussian blur to reduce noise
            img_enhanced = cv2.GaussianBlur(img_enhanced, (3, 3), 0.5)
            
            faces = self.mtcnn.detect(img_enhanced)
            embeddings = []

            if faces[0] is not None:
                boxes = faces[0]
                for box in boxes:
                    try:
                        box = box.astype(int)
                        face = img_enhanced[box[1]:box[3], box[0]:box[2]]
                        
                        # Validate face dimensions before processing
                        if face.shape[0] <= 5 or face.shape[1] <= 5 or face.shape[0] == 0 or face.shape[1] == 0:
                            continue
                        
                        # Apply additional preprocessing for better features
                        face_pil = Image.fromarray(face)
                        
                        # Convert to tensor with normalization
                        face_tensor = transforms.ToTensor()(face_pil)
                        face_tensor = transforms.Resize((160, 160))(face_tensor)
                        
                        # Apply normalization consistent with model training
                        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        face_tensor = normalize(face_tensor)
                        
                        face_tensor = face_tensor.unsqueeze(0).to(self.device)
                        with torch.no_grad(), autocast(enabled=(self.device.type=='cuda')):
                            embedding = self.model(face_tensor).cpu().numpy()[0]
                        embeddings.append(embedding)
                    except Exception as e:
                        self.log_status(f"Error processing face: {str(e)}")
                        continue
            return embeddings
        except Exception as e:
            self.log_status(f"Error processing image {image_path}: {str(e)}")
            return []

    def start_camera(self):
        self.cameras = []
        for source in self.camera_sources:
            try:
                camera = cv2.VideoCapture(source)
                if not camera.isOpened():
                    self.log_status(f"Warning: Failed to open camera source {source}")
                    camera = None
                else:
                    # Optimize video file reading
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for real-time processing
                    
                    # For video files, set faster reading
                    if source.endswith(('.mp4', '.avi', '.mov')):
                        # Set to process every n'th frame for smoother playback
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Lower resolution if needed
                        frame_width = 1280
                        frame_height = 720
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                self.cameras.append(camera)
            except Exception as e:
                self.log_status(f"Error opening camera {source}: {str(e)}")
                self.cameras.append(None)

        for i in range(len(self.cameras)):
            self.camera_frames[i] = None

        self.frame_queues = []
        self.camera_threads = []
        self.stop_threads = False

        for i in range(len(self.cameras)):
            queue = Queue(maxsize=2)
            self.frame_queues.append(queue)
            thread = threading.Thread(target=self.camera_thread, args=(i, queue), daemon=True)
            thread.start()
            self.camera_threads.append(thread)
            
        # Start processing the log queue from the main thread
        self.process_log_queue()
            
        self.update_video()

    def camera_thread(self, camera_idx, queue):
        camera = self.cameras[camera_idx]
        if camera is None:
            return
        consecutive_errors = 0
        last_frame_time = time.time()
        frame_skip = 0  # Skip counter for video files

        while not self.stop_threads:
            if not self.active_cameras[camera_idx]:
                time.sleep(0.1)  # Pause when camera inactive
                continue
                
            try:
                # For video files apply frame skipping for better  performance
                if self.camera_sources[camera_idx].endswith(('.mp4', '.avi', '.mov')):
                    if frame_skip > 0:
                        frame_skip -= 1
                        # Using grab/retrieve is faster than read()
                        ret = camera.grab()
                        if ret:
                             ret, frame = camera.retrieve()
                        else:
                             frame = None
                        if not ret:
                            consecutive_errors += 1
                            continue
                    else:
                        ret = camera.grab()
                        if ret:
                             ret, frame = camera.retrieve()
                        else:
                             frame = None
                        frame_skip = 2 
                else:
                    ret = camera.grab()
                    if ret:
                         ret, frame = camera.retrieve()
                    else:
                         frame = None
                    
                current_time = time.time()
                frame_interval = current_time - last_frame_time
                last_frame_time = current_time

                if ret and frame is not None and frame.size > 0:
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except:
                            break
                    frame_with_time = (frame, current_time)
                    queue.put(frame_with_time)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        self.log_status(f"Camera {camera_idx+1} multiple failures, reconnecting...")
                        camera.release()
                        time.sleep(0.5)
                        camera = cv2.VideoCapture(self.camera_sources[camera_idx])
                        if camera.isOpened():
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                            if self.camera_sources[camera_idx].endswith(('.mp4', '.avi', '.mov')):
                                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_width = 1280
                                frame_height = 720
                                camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                            self.cameras[camera_idx] = camera
                            self.log_status(f"Camera {camera_idx+1} reconnected")
                            consecutive_errors = 0
                        else:
                            self.log_status(f"Failed to reconnect camera {camera_idx+1}")
                target_interval = 1/20.0  # Cap at 20 FPS
                if frame_interval < target_interval:
                    time.sleep(max(0, target_interval - frame_interval))
            except Exception as e:
                consecutive_errors += 1
                self.log_status(f"Error in camera {camera_idx+1}: {str(e)}")
                time.sleep(0.1)

            # Small sleep to prevent high CPU usage in tight loops
            time.sleep(0.005)

    def update_video(self):
        """Updates video feeds from cameras and manages face detection/recognition processing."""
        process_time_start = time.time()
        
        # Monitor system load for adaptive processing
        current_cpu_usage = psutil.cpu_percent(interval=None)
        
        # Determine if this frame should be processed based on skip rate
        should_process = (self.frame_counter % self.detection_skip_rate == 0)

        # Update tracking timeouts for all tracked faces
        for track_id in list(self.face_tracking.keys()):
            self.face_tracking[track_id]['ttl'] -= 1
            if self.face_tracking[track_id]['ttl'] <= 0:
                # Keep recently disappeared faces a bit longer but mark as expired
                self.face_tracking[track_id]['ttl'] = -5
                # Clean up associated liveness data
                if track_id in self.liveness_history:
                    del self.liveness_history[track_id]
                if track_id in self.last_liveness_check:
                    del self.last_liveness_check[track_id]
        
        # Remove completely expired face tracks
        self.face_tracking = {k: v for k, v in self.face_tracking.items() if v['ttl'] > -30}
            
        # Process each active camera
        for i in range(len(self.cameras)):
            try:
                if self.active_cameras[i]:
                    if not self.frame_queues[i].empty():
                        try:
                            frame_with_time = self.frame_queues[i].get_nowait()
                            frame, frame_timestamp = frame_with_time
                            if frame is None or frame.size == 0:
                                continue
                                
                            # Store frame for display regardless of processing status
                            self.camera_frames[i] = frame.copy()
                            self.display_frames[i] = frame.copy()
                            
                            # Submit for processing if recognition is active and it's time to process
                            if self.recognition_active and should_process:
                                self.processing_executor.submit(
                                    self.process_frame,
                                    i, frame.copy(), frame_timestamp
                                )
                        except Exception as e:
                            pass
            except Exception as e:
                pass

        # Update display with the latest frames
        self.update_display()
            
        process_time_end = time.time()
        processing_duration = process_time_end - process_time_start
        self.last_processing_time = processing_duration

        # Dynamically adjust processing rate based on system load
        if self.adaptive_skip_rate:
            avg_face_count = sum(self.face_count_history) / max(1, len(self.face_count_history))
            
            # Increase skip rate if CPU usage is too high
            if current_cpu_usage > self.cpu_threshold:
                increase = min(self.max_cpu_skip_rate_increase, 
                              max(1, int((current_cpu_usage - self.cpu_threshold) / 5)))
                new_skip_rate = min(self.max_skip_rate, self.detection_skip_rate + increase)
                if new_skip_rate > self.detection_skip_rate:
                    self.detection_skip_rate = new_skip_rate
                    self.log_status(f"High CPU ({current_cpu_usage:.1f}%), increasing skip rate to {self.detection_skip_rate}")
            # Decrease skip rate when system load is low
            elif current_cpu_usage < self.cpu_threshold - 10 and processing_duration < 0.080 and avg_face_count < 5:
                 if self.detection_skip_rate > self.min_skip_rate:
                    self.detection_skip_rate = max(self.min_skip_rate, self.detection_skip_rate - 1)
            # Adjust based on processing time if CPU usage is acceptable
            elif processing_duration > 0.120 and self.detection_skip_rate < self.max_skip_rate:
                 self.detection_skip_rate += 1
            elif processing_duration < 0.080 and self.detection_skip_rate > self.min_skip_rate:
                 # Only decrease if we're not processing too many faces
                 if avg_face_count < 10:
                    self.detection_skip_rate = max(self.min_skip_rate, self.detection_skip_rate - 1)

            # Keep skip rate within allowed range
            self.detection_skip_rate = max(self.min_skip_rate, min(self.max_skip_rate, self.detection_skip_rate))


        # Calculate confidence scores periodically
        if self.recognition_active and self.frame_counter % self.frames_interval == self.frames_interval - 1:
            self.calculate_multi_camera_confidence()

        # Update frame counter
        self.frame_counter += 1
        if self.frame_counter >= self.frames_interval:
            self.frame_counter = 0
            
        # Schedule next frame update with fixed timings for smooth display
        update_delay = 33 
        self.root.after(update_delay, self.update_video)

    def recognize_face(self, face_embedding, box=None, camera_idx=0):
        """
        Matches a face embedding against known identities with temporal consistency.
        Returns the matched ID and similarity score.
        """
        matched_id = "Unknown"
        max_similarity = 0
        threshold = self.threshold_var.get()
        identity_key = None
        
        # Use slightly lower threshold for initial matches to catch borderline cases
        initial_threshold = max(0.40, threshold - 0.1)
        
        # Check if we can use face tracking for faster recognition
        if box is not None:
            # Calculate center point and dimensions
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            
            # Look for matching faces we're already tracking
            best_tracking_match = None
            best_tracking_iou = 0
            
            for track_id, track_info in self.face_tracking.items():
                if track_info['ttl'] <= 0 or track_info['camera_idx'] != camera_idx:
                    continue
                    
                last_box = track_info['box']
                last_center_x = (last_box[0] + last_box[2]) // 2
                last_center_y = (last_box[1] + last_box[3]) // 2
                
                # Check if centers are close enough
                distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                
                if distance < (box_width + last_box[2] - last_box[0]) * 0.3:
                    # Calculate overlap using Intersection over Union
                    x1 = max(box[0], last_box[0])
                    y1 = max(box[1], last_box[1])
                    x2 = min(box[2], last_box[2])
                    y2 = min(box[3], last_box[3])
                    
                    if x2 > x1 and y2 > y1:
                        overlap = (x2 - x1) * (y2 - y1)
                        box_area = box_width * box_height
                        last_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
                        iou = overlap / (box_area + last_area - overlap)
                        
                        if iou > best_tracking_iou:
                            best_tracking_iou = iou
                            best_tracking_match = track_id
            
            # If we found the same face with good overlap, use the cached identity
            if best_tracking_match and best_tracking_iou > 0.3:
                track_info = self.face_tracking[best_tracking_match]
                
                # Update tracking info
                track_info['box'] = box
                track_info['ttl'] = self.tracking_ttl
                track_info['camera_idx'] = camera_idx
                
                # Return the tracked identity if it's not unknown
                if track_info['identity'] != "Unknown":
                    return track_info['identity'], track_info['similarity']
                    
                # If it's unknown, we'll try to recognize it again
                identity_key = best_tracking_match
        
        # Perform face recognition using embeddings
        if face_embedding is not None:
            query = face_embedding.reshape(1, -1)
            candidate_matches = []
            
            for person_id, embeddings in self.known_faces.items():
                if not embeddings:
                    continue
                    
                if not isinstance(embeddings, np.ndarray):
                    embeddings_array = np.array(embeddings)
                    if embeddings_array.ndim == 1:
                        embeddings_array = embeddings_array.reshape(1, -1)
                else:
                    embeddings_array = embeddings
                    
                try:
                    # Calculate similarities for all embeddings of this person
                    similarities = cosine_similarity(query, embeddings_array)[0]
                    
                    # Get both max and top 3 weighted similarity
                    max_sim = np.max(similarities)
                    
                    # Take average of top 3 similarities if available
                    if len(similarities) >= 3:
                        # Sort similarities in descending order
                        top_similarities = np.sort(similarities)[-3:]
                        
                        # Weight the top matches more heavily (40%, 30%, 30%)
                        weighted_similarity = (
                            top_similarities[2] * 0.4 + 
                            top_similarities[1] * 0.3 + 
                            top_similarities[0] * 0.3
                        )
                    elif len(similarities) == 2:
                        # Weight 60/40 for two matches
                        top_similarities = np.sort(similarities)[-2:]
                        weighted_similarity = top_similarities[1] * 0.6 + top_similarities[0] * 0.4
                    else:
                        weighted_similarity = max_sim
                        
                    # Use max for quick filtering, then add detail
                    if max_sim > initial_threshold:
                        candidate_matches.append((person_id, weighted_similarity, max_sim))
                except Exception as e:
                    continue
            
            # Sort candidates by weighted similarity
            if candidate_matches:
                candidate_matches.sort(key=lambda x: x[1], reverse=True)
                best_match, best_similarity, best_max = candidate_matches[0]
                
                # Use the best match if it's above threshold
                if best_similarity > threshold:
                    matched_id = best_match
                    max_similarity = best_similarity
                    if matched_id not in self.attendance:
                        self.update_attendance(matched_id)
                # If there's a significant gap between best and second best, use the best match
                elif len(candidate_matches) >= 2:
                    second_best = candidate_matches[1][1]
                    # More aggressive confidence gap (0.08 instead of 0.1)
                    if best_similarity > initial_threshold and (best_similarity - second_best) > 0.08:
                        matched_id = best_match
                        max_similarity = best_similarity
                        if matched_id not in self.attendance:
                            self.update_attendance(matched_id)
                # If max similarity is strong, use it (helps with strong single feature matches)
                elif best_max > threshold + 0.05:
                    matched_id = best_match
                    max_similarity = best_max
                    if matched_id not in self.attendance:
                        self.update_attendance(matched_id)
                
        # Update or create tracking
        if box is not None:
            if identity_key is None:
                # Create a new tracking entry
                identity_key = f"track_{camera_idx}_{int(time.time())}_{len(self.face_tracking)}"
                
            # Store or update tracking
            self.face_tracking[identity_key] = {
                'identity': matched_id,
                'similarity': max_similarity,
                'box': box,
                'ttl': self.tracking_ttl,
                'camera_idx': camera_idx
            }
                
        return matched_id, max_similarity

    def update_attendance(self, person_id):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if person_id not in self.attendance:
            student_info = self.student_data.get(person_id, {})
            if not student_info:
                self.log_status(f"Student ID {person_id} not found in attendance sheet")
                return
            self.active_tracking.add(person_id)
            self.attendance[person_id] = {
                'ID': person_id,
                'Name': student_info['name'],
                'Status': "Processing...",
                'Time': timestamp,
                'Confidence': "Processing..."
            }
            self.attendance_tree.insert('', 'end', values=(
                person_id,
                student_info['name'],
                "Processing...",
                timestamp,
                "Processing..."
            ))
            self.log_status(f"Started tracking {student_info['name']} ({person_id})")
            
            # Update present/absent counts
            self.present_students_var.set(str(len(self.active_tracking)))
            self.absent_students_var.set(str(len(self.student_data) - len(self.active_tracking)))

    def export_attendance(self):
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            master_df = pd.read_excel(self.attendance_file)
            attendance_records = []
            for _, student in master_df.iterrows():
                student_id = str(student['ID'])
                record = {
                    'ID': student_id,
                    'Name': student['Name'],
                    'Status': 'Absent',
                    'Time': 'N/A',
                    'Date': current_date,
                    'Confidence': '0.0%',
                    'Liveness': 'N/A'
                }
                if student_id in self.attendance:
                    # Check liveness status
                    liveness_status = "Passed"
                    if "Failed liveness check" in self.attendance[student_id]['Status']:
                        liveness_status = "Failed"
                    
                    record.update({
                        'Status': self.attendance[student_id]['Status'],
                        'Time': self.attendance[student_id]['Time'],
                        'Confidence': self.attendance[student_id]['Confidence'],
                        'Liveness': liveness_status
                    })
                attendance_records.append(record)
            current_df = pd.DataFrame(attendance_records)
            export_path = os.path.join(os.path.dirname(self.attendance_file), f'Attendance_{current_date}.xlsx')
            current_df.to_excel(export_path, index=False)
            self.log_status(f"Attendance exported to {export_path}")
            
            # Calculate attendance statistics
            total_students = len(master_df)
            present_count = sum(1 for record in attendance_records if record['Status'].startswith('Present'))
            absent_count = total_students - present_count
            failed_liveness_count = sum(1 for record in attendance_records if "Failed liveness check" in record['Status'])
            unknown_count = int(self.unknown_faces_var.get())
            summary = f"""ATTENDANCE SUMMARY FOR {current_date}
------------------------------------------
REGISTRATION STATISTICS:
Total Registered Students: {total_students}
Present: {present_count} ({(present_count/total_students*100) if total_students > 0 else 0:.1f}%)
Absent: {absent_count} ({(absent_count/total_students*100) if total_students > 0 else 0:.1f}%)

ADVANCED STATISTICS:
Unknown Faces Detected: {unknown_count}
Failed Liveness Checks: {failed_liveness_count}
Total People Detected: {present_count + unknown_count}

DETAILED STATUS:"""
            for record in attendance_records:
                status_icon = "✓" if record['Status'].startswith('Present') else "✗"
                confidence_note = ""
                if record['Status'].startswith('Present'):
                    confidence_note = f" - Confidence: {record['Confidence']}"
                    if record['Liveness'] == "Failed":
                        confidence_note += " (FAKE FACE DETECTED)"
                summary += f"\n{status_icon} {record['Name']} ({record['ID']}): {record['Status']}{confidence_note}"
            messagebox.showinfo("Export Success", summary)
            
            # Save summary to text file alongside Excel export
            summary_path = os.path.join(os.path.dirname(self.attendance_file), f'Summary_{current_date}.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)
            self.log_status(f"Summary exported to {summary_path}")
            
        except Exception as e:
            self.log_status(f"Error exporting attendance: {str(e)}")
            messagebox.showerror("Error", f"Failed to export attendance: {str(e)}")

    def clear_attendance(self):
        try:
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            self.attendance = {}
            self.face_confidence = {}
            self.confidence_history = {}
            self.active_tracking = set()
            self.frame_counter = 0
            self.liveness_history = {}
            
            # Reset counters
            total_count = len(self.student_data)
            self.present_students_var.set("0")
            self.absent_students_var.set(str(total_count))
            self.unknown_faces_var.set("0")
            self.fake_faces_var.set("0")
            
            # Reset summary section
            self.summary_total_var.set(str(total_count))
            self.summary_present_var.set("0 (0.0%)")
            self.summary_absent_var.set(f"{total_count} (100.0%)")
            self.summary_rate_var.set("0.0%")
            self.summary_unknown_var.set("0")
            self.summary_fake_var.set("0")
            self.summary_detected_var.set("0")
            
            # Reset the attendance rate bar
            width = self.rate_canvas.winfo_width()
            if width > 0:
                self.rate_canvas.delete("all")
                self.rate_canvas.create_rectangle(0, 0, width, 20, fill='#FF5252', outline='')
            
            self.status_text.delete(1.0, tk.END)
            self.log_status("Attendance cleared")
        except Exception as e:
            self.log_status(f"Error clearing attendance: {str(e)}")

    def mark_attendance(self):
        pass

    def view_logs(self):
        pass

    def log_status(self, message):
        """Adds a timestamped message to the log queue for display in the UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_queue.put(log_entry)

    def process_log_queue(self):
        """Processes queued log messages and updates the UI with appropriate formatting"""
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                
                # Add message to log display
                self.status_text.insert(tk.END, message + "\n")
                
                # Apply color formatting based on message type
                if "Error" in message or "FAKE DETECTED" in message:
                    self.status_text.tag_add("error", "end-2c linestart", "end-1c")
                    self.status_text.tag_config("error", foreground="#ff6b6b") # Red
                elif "Warning" in message:
                    self.status_text.tag_add("warning", "end-2c linestart", "end-1c")
                    self.status_text.tag_config("warning", foreground="#ffd93d") # Yellow/Orange
                elif "Success" in message or "Present" in message or "started" in message or "reconnected" in message:
                     self.status_text.tag_add("success", "end-2c linestart", "end-1c")
                     self.status_text.tag_config("success", foreground="#6bff6b") # Green
                elif "High CPU" in message or "increasing skip rate" in message:
                     self.status_text.tag_add("info", "end-2c linestart", "end-1c")
                     self.status_text.tag_config("info", foreground="#A9A9A9") # Gray for info
                
                # Auto-scroll to latest message
                self.status_text.see(tk.END)
                
        except Exception as e:
            # Avoid crashing the UI if log processing fails
            print(f"Error processing log queue: {e}")
            pass

        # Check for new logs every 100ms
        self.root.after(100, self.process_log_queue)

    def exit_app(self):
        self.stop_threads = True
        for thread in self.camera_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        for camera in self.cameras:
            if camera and camera.isOpened():
                camera.release()
        self.root.quit()

    def run(self):
        self.root.mainloop()

    def start_recognition(self):
        self.recognition_active = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log_status("Face recognition started")

    def stop_recognition(self):
        self.recognition_active = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Clear all display annotations when stopping recognition
        self.display_annotations = {}
        
        # Collect session statistics
        present_count = 0
        fake_faces_count = 0
        
        for person_id in self.active_tracking:
            if person_id in self.confidence_history and self.confidence_history[person_id]:
                confidence_values = self.confidence_history[person_id]
                avg_confidence = sum(confidence_values) / len(confidence_values)
                
                # Check if this person's face passed liveness detection consistently
                is_live_person = True
                
                # Check all tracking IDs that might contain this person
                for track_id, track_info in self.face_tracking.items():
                    if track_info['identity'] == person_id:
                        # Check for liveness in tracked face
                        if track_id in self.liveness_history:
                            liveness = self.liveness_history[track_id]
                            if not liveness.get('live', True):
                                # If any tracking instance is marked as fake, flag it
                                is_live_person = False
                                self.log_status(f"Warning: {person_id} may be using a photo - failed liveness check")
                                fake_faces_count += 1
                                break
                
                # Adjust confidence if person may be using a photo
                if not is_live_person:
                    avg_confidence = max(0, avg_confidence - 40)  # Heavily penalize non-live faces
                
                avg_confidence_str = f"{avg_confidence:.1f}%"
                status = "Present" if avg_confidence >= 50.0 and is_live_person else "Absent"
                
                # If marked absent due to liveness check, note it
                status_suffix = ""
                if avg_confidence >= 50.0 and not is_live_person:
                    status_suffix = " (Failed liveness check)"
                
                try:
                    for item in self.attendance_tree.get_children():
                        current_values = self.attendance_tree.item(item)['values']
                        if str(current_values[0]) == str(person_id):
                            new_values = [
                                current_values[0],
                                current_values[1],
                                status + status_suffix,
                                current_values[3],
                                avg_confidence_str
                            ]
                            self.attendance_tree.item(item, values=new_values)
                            self.log_status(f"Updated table entry - ID: {person_id}, Status: {status}{status_suffix}, Confidence: {avg_confidence_str}")
                            break
                except Exception as e:
                    self.log_status(f"Error updating GUI table: {str(e)}")
                    
                if person_id in self.attendance:
                    self.attendance[person_id].update({
                        'Status': status + status_suffix,
                        'Confidence': avg_confidence_str
                    })
                    
                student_name = self.attendance[person_id]['Name']
                final_status = f"Final Status for {student_name} ({person_id}): {status}{status_suffix} with average confidence: "
                self.status_text.insert(tk.END, final_status)
                self.status_text.insert(tk.END, avg_confidence_str + "\n")
                last_line_start = self.status_text.index("end-2c linestart")
                last_line_end = self.status_text.index("end-1c")
                confidence_start = f"{last_line_start} + {len(final_status)}c"
                self.status_text.tag_add("green", confidence_start, last_line_end)
                self.status_text.tag_config("green", foreground="#6bff6b")
                self.root.update_idletasks()
                
                if status == "Present":
                    present_count += 1
                
        # Update present/absent counts in the UI
        total_students = len(self.student_data)
        absent_count = total_students - present_count
        self.present_students_var.set(str(present_count))
        self.absent_students_var.set(str(absent_count))
        
        # Update other statistics
        unknown_count = int(self.unknown_faces_var.get())
        self.fake_faces_var.set(str(fake_faces_count))
        
        # Calculate attendance rate
        attendance_rate = (present_count / total_students) * 100 if total_students > 0 else 0
        self.attendance_rate_var.set(f"{attendance_rate:.1f}%")
        
        # Display session summary
        session_summary = f"""
RECOGNITION SESSION SUMMARY
--------------------------
Total Students: {total_students}
Present: {present_count} ({attendance_rate:.1f}%)
Absent: {absent_count} ({100-attendance_rate:.1f}%)
Unknown Faces: {unknown_count}
Fake Faces Detected: {fake_faces_count}
"""
        self.log_status(session_summary)
        
        # Update the summary section in the GUI
        self.update_summary_section(total_students, present_count, absent_count, 
                                  unknown_count, fake_faces_count, attendance_rate)
                
        self.confidence_history = {}
        self.active_tracking = set()
        self.face_confidence = {}
        self.liveness_history = {}
        
        self.log_status("Face recognition stopped")
        
    def update_summary_section(self, total, present, absent, unknown, fake, rate):
        """Update the summary section in the left panel with the latest statistics"""
        # Update student statistics
        self.summary_total_var.set(str(total))
        self.summary_present_var.set(f"{present} ({rate:.1f}%)")
        self.summary_absent_var.set(f"{absent} ({100-rate:.1f}%)")
        self.summary_rate_var.set(f"{rate:.1f}%")
        
        # Update face detection statistics
        self.summary_unknown_var.set(str(unknown))
        self.summary_fake_var.set(str(fake))
        # Calculate total detected based on the number of students in the attendance dictionary
        total_detected = len(self.attendance) 
        self.summary_detected_var.set(str(total_detected))
        
        # Update attendance rate bar
        self.rate_canvas.delete("all")
        width = self.rate_canvas.winfo_width()
        if width < 10:  # Handle initial render when width not yet available
            width = 300
        
        bar_width = int(width * (rate/100))
        self.rate_canvas.create_rectangle(0, 0, bar_width, 20, fill='#4CAF50', outline='')
        self.rate_canvas.create_rectangle(bar_width, 0, width, 20, fill='#FF5252', outline='')
        
        # Add text to bar
        percentage_text = f"{rate:.1f}%"
        text_x = min(bar_width - 5, width // 2)
        text_x = max(text_x, width // 2)
        self.rate_canvas.create_text(text_x, 10, text=percentage_text, fill='white', font=("Arial", 9, "bold"))

    def update_face_tracking(self, box, person_id, similarity, camera_idx):
        """Update tracking for a recognized face"""
        # Check if this matches a face we're already tracking
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        best_tracking_match = None
        best_tracking_iou = 0
        
        for track_id, track_info in self.face_tracking.items():
            if track_info['ttl'] <= 0 or track_info['camera_idx'] != camera_idx:
                continue
                
            last_box = track_info['box']
            last_center_x = (last_box[0] + last_box[2]) // 2
            last_center_y = (last_box[1] + last_box[3]) // 2
            
            # Calculate distance
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            
            # If centers are close enough, consider it the same face
            if distance < (box_width + last_box[2] - last_box[0]) * 0.3:
                x1 = max(box[0], last_box[0])
                y1 = max(box[1], last_box[1])
                x2 = min(box[2], last_box[2])
                y2 = min(box[3], last_box[3])
                
                if x2 > x1 and y2 > y1:
                    overlap = (x2 - x1) * (y2 - y1)
                    box_area = box_width * box_height
                    last_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
                    iou = overlap / (box_area + last_area - overlap)
                    
                    if iou > best_tracking_iou:
                        best_tracking_iou = iou
                        best_tracking_match = track_id
        
        # If we found a good match in tracking, update it
        if best_tracking_match and best_tracking_iou > 0.3:
            # Update tracking with new info
            track_info = self.face_tracking[best_tracking_match]
            
            # If this is not "Unknown" and the previous was, update it
            if track_info['identity'] == "Unknown" and person_id != "Unknown":
                track_info['identity'] = person_id
                track_info['similarity'] = similarity
            
            # Update position and TTL
            track_info['box'] = box
            track_info['ttl'] = self.tracking_ttl
            
        else:
            # Create a new tracking entry
            identity_key = f"track_{camera_idx}_{int(time.time())}_{len(self.face_tracking)}"
            self.face_tracking[identity_key] = {
                'identity': person_id,
                'similarity': similarity,
                'box': box,
                'ttl': self.tracking_ttl,
                'camera_idx': camera_idx
            }

    def update_display(self):
        """Update the display with the latest frames, independent of processing"""
        # Count unique unknown faces across all cameras
        total_unknown = 0
        total_fake_faces = 0
        
        for track_id, track_info in self.face_tracking.items():
            # Count unknown faces
            if track_info['identity'] == "Unknown" and track_info['ttl'] > 0:
                total_unknown += 1
            
            # Count fake faces (from liveness detection)
            if track_id in self.liveness_history and not self.liveness_history[track_id].get('live', True) and track_info['ttl'] > 0:
                total_fake_faces += 1
                
        # Update statistics
        self.unknown_faces_var.set(str(total_unknown))
        self.fake_faces_var.set(str(total_fake_faces))
        
        # Calculate and update attendance rate
        total_students = int(self.total_students_var.get()) or 1
        present_students = int(self.present_students_var.get()) or 0
        attendance_rate = (present_students / total_students) * 100
        self.attendance_rate_var.set(f"{attendance_rate:.1f}%")
        
        # Update summary section as well if recognition is active
        if self.recognition_active:
            absent_students = total_students - present_students
            self.update_summary_section(total_students, present_students, absent_students, 
                                      total_unknown, total_fake_faces, attendance_rate)
                
        for i in range(len(self.cameras)):
            try:
                if self.active_cameras[i]:
                    display_frame = self.display_frames.get(i)
                    if display_frame is None:
                        continue
                    
                    # Get annotations if available - only when recognition is active
                    annotations = self.display_annotations.get(i, []) if self.recognition_active else []
                    
                    # Draw annotations on a copy to avoid modifying the original
                    display_copy = display_frame.copy()
                    
                    # Draw all annotations
                    for det in annotations:
                        # Check if liveness is available in the detection
                        is_live = det.get('is_live', True)  # Default to True if not specified
                        liveness_score = det.get('liveness', 1.0)  # Default to 1.0 if not specified
                        
                        # Choose color based on identity and liveness
                        if det['matched_id'] != "Unknown":
                            if is_live:
                                color = (0, 255, 0)  # Green for known and live
                            else:
                                color = (0, 165, 255)  # Orange for known but not live (likely photo)
                        else:
                            if is_live:
                                color = (0, 0, 255)  # Red for unknown but live
                            else:
                                color = (128, 128, 128)  # Gray for unknown and not live
                                
                        cv2.rectangle(display_copy, 
                                     (det['box'][0], det['box'][1]), 
                                     (det['box'][2], det['box'][3]), 
                                     color, 2)
                        
                        # Add liveness indicator if not live
                        display_text = det['matched_id']
                        if not is_live:
                            display_text += " (FAKE)"
                        
                        cv2.putText(display_copy, 
                                   f"{display_text} ({det['similarity']:.2f})", 
                                   (det['box'][0], det['box'][1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Add liveness score below
                        cv2.putText(display_copy,
                                  f"Live: {liveness_score:.2f}",
                                  (det['box'][0], det['box'][3] + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Convert frame for display
                    try:
                        display_rgb = cv2.cvtColor(display_copy, cv2.COLOR_BGR2RGB)
                        if i == 0:
                            disp_w, disp_h = 640, 480
                            interpolation = cv2.INTER_AREA # Keep better quality for main display
                        else:
                            disp_w, disp_h = 320, 240
                            interpolation = cv2.INTER_LINEAR # FASTER: Use linear for smaller displays
                        resized = cv2.resize(display_rgb, (disp_w, disp_h), interpolation=interpolation)
                        img = Image.fromarray(resized)
                        photo = ImageTk.PhotoImage(image=img)
                        self.video_labels[i].configure(image=photo, background='#f0f0f0')
                        self.video_labels[i].image = photo
                    except Exception as e:
                        pass
                else:
                    self.video_labels[i].configure(image='', text="Camera Off", background='#f0f0f0')
            except Exception as e:
                pass

    def process_frame(self, camera_idx, frame, frame_timestamp):
        """Process a frame for face detection and recognition using ThreadPoolExecutor"""
        try:
            # Basic RGB conversion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Adaptive preprocessing based on recent face counts
            avg_face_count = sum(self.face_count_history) / max(1, len(self.face_count_history))
            process_start_time = time.time()

            if avg_face_count > 8:
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                gray = cv2.equalizeHist(gray)
                frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                # Keep fuller preprocessing for fewer faces
                img_lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
                frame_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
                frame_rgb = cv2.GaussianBlur(frame_rgb, (3, 3), 0.5)

            # Prepare to accumulate detection results
            camera_detections = set()
            detections_cached = []
            detections_batch = []
            all_detections = []

            # Face detection
            boxes_combined = []
            probs_combined = []
            try:
                h, w = frame.shape[:2]
                faces = self.mtcnn.detect(Image.fromarray(frame_rgb))

                if faces[0] is not None:
                    boxes_combined = faces[0] # Bounding boxes
                    probs_combined = faces[1] # Detection probabilities

            except Exception as detect_error:
                self.log_status(f"MTCNN detection error on camera {camera_idx}: {detect_error}")
                boxes_combined = [] # Ensure lists are empty on error
                probs_combined = []


            # Process detected boxes
            if len(boxes_combined) > 0:
                # Filter low probability detections FIRST
                min_detection_probability = 0.88
                valid_indices = [i for i, prob in enumerate(probs_combined) if prob >= min_detection_probability]
                
                if not valid_indices: # Exit early if no faces pass probability threshold
                    with self.processing_lock: # Need lock to update annotations
                         self.display_annotations[camera_idx] = []
                    return

                boxes_combined = boxes_combined[valid_indices]
                probs_combined = probs_combined[valid_indices] # Keep corresponding probabilities if needed

                # Limit number of faces based on probability/size before NMS
                face_count = len(boxes_combined)
                self.face_count_history.append(face_count)
                self.face_count_history.pop(0)

                if face_count > self.max_faces_per_frame:
                    # Sort by probability (higher first) or size if needed
                    sorted_indices = np.argsort(probs_combined)[::-1] # Sort by prob descending
                    boxes_combined = boxes_combined[sorted_indices[:self.max_faces_per_frame]]
                    probs_combined = probs_combined[sorted_indices[:self.max_faces_per_frame]] # Keep corresponding probs

                nms_threshold = 0.4
                def iou(box1, box2):
                    # Ensure box coordinates are valid numbers
                    if any(coord is None or not np.isfinite(coord) for coord in box1) or \
                       any(coord is None or not np.isfinite(coord) for coord in box2):
                        return 0.0
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    if x2 < x1 or y2 < y1: return 0.0
                    intersection = (x2 - x1) * (y2 - y1)
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    # Prevent division by zero or negative areas
                    if box1_area <= 0 or box2_area <= 0: return 0.0
                    union = box1_area + box2_area - intersection
                    return intersection / union if union > 0 else 0.0

                filtered_boxes = []
                # Use probabilities for sorting before NMS
                indices = np.argsort(probs_combined)[::-1] # Higher probability first
                while len(indices) > 0:
                    idx = indices[0]
                    current_box = boxes_combined[idx]
                    # Ensure current_box is valid before appending and using in iou
                    if current_box is None or len(current_box) != 4 or any(c is None or not np.isfinite(c) for c in current_box):
                         indices = indices[1:] # Skip invalid box
                         continue
                    filtered_boxes.append(current_box)
                    # Remove overlapping boxes with lower probability
                    remaining_indices = []
                    for i in indices[1:]:
                         # Ensure comparison box is also valid
                         compare_box = boxes_combined[i]
                         if compare_box is None or len(compare_box) != 4 or any(c is None or not np.isfinite(c) for c in compare_box):
                            continue # Skip invalid comparison box
                         if iou(current_box, compare_box) < nms_threshold:
                             remaining_indices.append(i)
                    indices = np.array(remaining_indices) # Convert back to numpy array if needed

                boxes = np.array(filtered_boxes)

                # Limit again after NMS just in case
                if len(boxes) > self.max_faces_per_frame:
                     boxes = boxes[:self.max_faces_per_frame]

                current_time = time.time()
                for box in boxes:
                    try:
                        if box is None or len(box) != 4 or any(c is None or not np.isfinite(c) for c in box):
                             continue
                        box = box.astype(int)
                        x1, y1, x2, y2 = box
                        frame_height, frame_width = frame.shape[:2]
                        # Add boundary clipping
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)

                        width = x2 - x1
                        height = y2 - y1

                        # Skip tiny faces more aggressively
                        # Use self.mtcnn.min_face_size which was increased earlier
                        if width <= 0 or height <= 0 or width < self.mtcnn.min_face_size or height < self.mtcnn.min_face_size:
                            continue

                        cache_key = f"{camera_idx}_{x1//20}_{y1//20}_{width//20}_{height//20}"
                        box_tuple = (x1, y1, x2, y2)

                        # Check for tracking match first
                        track_id, track_sim = self.check_face_tracking(box_tuple, camera_idx)

                        # Initialize liveness info
                        liveness_score = 1.0 # Default to live
                        is_live = True
                        should_check_liveness = False

                        # Determine if liveness check is needed
                        if self.liveness_detection_enabled:
                            if track_id != "Unknown":
                                last_check_time = self.last_liveness_check.get(track_id, 0)
                                if current_time - last_check_time > self.liveness_check_interval:
                                    should_check_liveness = True
                            else:
                                # Always check liveness for potentially unknown faces
                                should_check_liveness = True

                        # Extract face region only if needed for liveness or batch processing
                        face_region = None
                        if should_check_liveness or track_id == "Unknown":
                             # Use original frame (BGR) for liveness/recognition if possible, less preprocessing artifacts
                             face_region = frame[y1:y2, x1:x2] # Use BGR frame for dlib/cv2 liveness checks
                             if face_region is None or face_region.size == 0 or face_region.shape[0] <= 0 or face_region.shape[1] <= 0:
                                 continue

                        # Perform liveness check if needed
                        if should_check_liveness and face_region is not None:
                            # Pass BGR face region to liveness check
                            liveness_score = self.check_liveness(face_region, box_tuple, camera_idx, track_id if track_id != "Unknown" else None)
                            is_live = liveness_score > self.liveness_threshold
                            if track_id != "Unknown":
                                self.last_liveness_check[track_id] = current_time # Update last check time
                            # Log fake detection immediately
                            if not is_live and track_id != "Unknown":
                                self.log_status(f"FAKE DETECTED for {track_id} (Score: {liveness_score:.2f})")
                            elif not is_live:
                                self.log_status(f"FAKE DETECTED for Unknown face at {box_tuple} (Score: {liveness_score:.2f})")


                        # Handle tracked face
                        if track_id != "Unknown":
                            detections_cached.append({
                                'box': box_tuple,
                                'matched_id': track_id,
                                'similarity': track_sim,
                                'cache_key': cache_key, # Still useful if tracking lost
                                'liveness': liveness_score,
                                'is_live': is_live
                            })
                            if is_live: camera_detections.add(track_id)
                            continue # Skip embedding extraction for tracked faces

                        # Handle potentially new/unknown face (check cache)
                        matched_id = "Unknown"
                        similarity = 0
                        cached_embedding = None
                        if (cache_key in self.embeddings_cache and
                            current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl):
                            cached_embedding, matched_id, similarity = self.embeddings_cache[cache_key]
                            # Use cached liveness if available and recent enough? Maybe not reliable.
                            # Re-check liveness for cached matches too if needed
                            detections_cached.append({
                                'box': box_tuple,
                                'matched_id': matched_id,
                                'similarity': similarity,
                                'cache_key': cache_key,
                                'liveness': liveness_score, # Use recently calculated score
                                'is_live': is_live
                            })
                            if matched_id != "Unknown":
                                self.update_face_tracking(box_tuple, matched_id, similarity, camera_idx)
                                if is_live: camera_detections.add(matched_id)
                        else:
                            # Prepare face region for embedding extraction (needs RGB)
                            if face_region is not None: # Should have been extracted above
                                face_rgb_for_model = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                                # Ensure valid shape before appending
                                if face_rgb_for_model.shape[0] > 0 and face_rgb_for_model.shape[1] > 0:
                                    # Append to batch for inference
                                    detections_batch.append({
                                        'box': box_tuple,
                                        'cache_key': cache_key,
                                        'face_region': face_rgb_for_model, # Pass RGB version
                                        'camera_idx': camera_idx,
                                        'liveness': liveness_score, # Pass calculated liveness
                                        'is_live': is_live
                                    })
                    except Exception as e:
                        self.log_status(f"Error processing box [{box}]: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Process next box

                # Process batch detections together
                if detections_batch:
                    max_batch_size = 8
                    for batch_start in range(0, len(detections_batch), max_batch_size):
                        batch_end = min(batch_start + max_batch_size, len(detections_batch))
                        current_batch = detections_batch[batch_start:batch_end]

                        face_tensors = []
                        valid_indices = []

                        for idx, det in enumerate(current_batch):
                            try:
                                # face_region is already RGB here
                                face_pil = Image.fromarray(det['face_region'])
                                if face_pil.size[0] <= 0 or face_pil.size[1] <= 0: continue

                                resample_method = Image.NEAREST if avg_face_count > 10 else Image.LANCZOS
                                face_pil = face_pil.resize((160, 160), resample_method)
                                face_tensor = transforms.ToTensor()(face_pil)
                                normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                face_tensor = normalize(face_tensor)
                                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                                face_tensors.append(face_tensor)
                                valid_indices.append(batch_start + idx)
                            except Exception as e:
                                self.log_status(f"Error preparing tensor for batch: {e}")
                                continue

                        if face_tensors:
                            try:
                                batch_tensor = torch.cat(face_tensors, dim=0)
                                with torch.no_grad(), autocast(enabled=(self.device.type=='cuda')):
                                    embeddings_batch = self.model(batch_tensor).cpu().numpy()

                                for batch_idx, det_idx in enumerate(valid_indices):
                                    if batch_idx >= len(embeddings_batch): continue
                                    embedding = embeddings_batch[batch_idx]
                                    det = detections_batch[det_idx]
                                    # Pass box tuple to recognize_face
                                    rec_id, sim = self.recognize_face(embedding, det['box'], camera_idx)
                                    det['matched_id'] = rec_id
                                    det['similarity'] = sim
                                    if rec_id != "Unknown":
                                        self.update_face_tracking(det['box'], rec_id, sim, camera_idx)
                                        # Only add to detections if live
                                        if det['is_live']:
                                            camera_detections.add(rec_id)
                                    # Update cache only if liveness check passed
                                    # Cache even if not live, but recognition uses live status
                                    self.embeddings_cache[det['cache_key']] = (embedding, rec_id, sim)
                                    self.cache_timestamps[det['cache_key']] = current_time
                            except Exception as e:
                                self.log_status(f"Batch processing error: {str(e)}")
                                pass

                # Combine cached and batch results
                all_detections = detections_cached + [d for d in detections_batch if 'matched_id' in d] # Ensure batch items were processed

                # Update per-camera detection tracking (only for live faces)
                for person_id in self.active_tracking:
                    camera_key = f"{camera_idx}_{person_id}"
                    if camera_key not in self.face_detections:
                        self.face_detections[camera_key] = [0] * self.frames_interval
                    
                    # Check if this person was detected *and* marked as live in this frame
                    detected_live = False
                    for det in all_detections:
                         # Match ID and ensure it's considered live
                        if det['matched_id'] == person_id and det.get('is_live', False):
                            detected_live = True
                            break
                            
                    current_idx = self.frame_counter % self.frames_interval
                    self.face_detections[camera_key][current_idx] = 1 if detected_live else 0


            # Store annotations for display (Needs lock)
            with self.processing_lock:
                self.display_annotations[camera_idx] = all_detections

            # Log processing time for this frame
            process_end_time = time.time()
            proc_duration = process_end_time - process_start_time
            # Log only if duration is significant to avoid spamming logs
            if proc_duration > 0.01:
                 self.log_status(f"Processed Cam {camera_idx} in {proc_duration:.4f}s ({len(all_detections)} faces shown)")

        except Exception as e:
            self.log_status(f"Error in process_frame thread for camera {camera_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure annotations are cleared or handled on error (Needs lock)
            with self.processing_lock:
                 self.display_annotations[camera_idx] = [] # Clear annotations on error

    def check_liveness(self, face_img, box, camera_idx, track_id=None):
        """
        Determines if a detected face is a real person or a photo/printout.
        Returns a score from 0-1, where scores above 0.5 indicate a likely real face.
        """
        if not self.liveness_detection_enabled:
            return 1.0  # Skip checks if detection is disabled
            
        liveness_score = 0.0
        h, w = face_img.shape[:2]
        
        # Skip faces that are too small for reliable analysis
        if h < 80 or w < 80:
            return 0.7  # Assign medium confidence to small faces
            
        try:
            # Analyze texture patterns - real faces have more variation
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            texture_score = self.check_texture(gray)
            
            # Check for eye blinks
            blink_score = self.check_eye_blink(face_img, box)
            
            # Check for subtle movements
            movement_score = 0.5  # Neutral starting value
            if track_id and track_id in self.face_tracking:
                movement_score = self.check_face_movement(track_id, box, camera_idx)
            
            # Combined weighted score
            liveness_score = (texture_score * 0.5) + (blink_score * 0.3) + (movement_score * 0.2)
            
            # Track scores over time for more stability
            if track_id:
                if track_id not in self.liveness_history:
                    self.liveness_history[track_id] = {
                        'scores': [],
                        'ttl': self.liveness_ttl,
                        'live': False
                    }
                
                history = self.liveness_history[track_id]
                history['scores'].append(liveness_score)
                history['ttl'] = self.liveness_ttl
                
                # Keep most recent scores
                if len(history['scores']) > 10:
                    history['scores'] = history['scores'][-10:]
                    
                # Average for stability
                avg_score = sum(history['scores']) / len(history['scores'])
                history['live'] = avg_score > self.liveness_threshold
                liveness_score = avg_score
        
        except Exception as e:
            # Default to medium confidence if analysis fails
            liveness_score = 0.6
            
        return liveness_score
    
    def check_texture(self, gray_face):
        """Analyzes texture patterns to distinguish real faces from flat images/printouts"""
        try:
            # Use gradient analysis to detect natural skin texture
            sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Photos tend to have less texture variation
            avg_magnitude = np.mean(magnitude)
            normalized_score = min(1.0, avg_magnitude / 20.0)  # Normalize to 0-1
            
            # Check local variance in small blocks
            block_size = 8
            h, w = gray_face.shape
            variances = []
            
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = gray_face[i:i+block_size, j:j+block_size]
                    variances.append(np.var(block))
            
            if variances:
                var_score = min(1.0, np.mean(variances) / 500.0)  # Normalize
                
                # Combine both texture metrics
                texture_score = (normalized_score + var_score) / 2
                return texture_score
            else:
                return normalized_score
                
        except Exception as e:
            return 0.5  # Neutral score if analysis fails
    
    def check_eye_blink(self, face_img, box):
        """Detects eye blinks as an indicator of a live person"""
        if not hasattr(self, 'landmark_predictor'):
            return 0.5  # Neutral score if predictor isn't available
            
        try:
            # Set up dlib rectangle for landmark detection
            x1, y1, x2, y2 = [int(c) for c in box]
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Get eye landmark points
            # Using 68-point model: Left eye: 36-41, Right eye: 42-47
            left_eye = []
            right_eye = []
            
            for i in range(36, 42):  # Left eye landmarks
                x, y = landmarks.part(i).x, landmarks.part(i).y
                left_eye.append((x, y))
                
            for i in range(42, 48):  # Right eye landmarks
                x, y = landmarks.part(i).x, landmarks.part(i).y
                right_eye.append((x, y))
            
            # Calculate eye aspect ratio (height/width ratio)
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Low ratio indicates closed eyes (blink detected)
            if ear < self.blink_threshold:
                return 0.9  # High liveness score when blink detected
            else:
                return 0.6  # Medium score for open eyes
                
        except Exception as e:
            return 0.5  # Neutral score if detection fails
    
    def eye_aspect_ratio(self, eye):
        """Calculates eye aspect ratio to detect blinks"""
        # Vertical distances between eye landmarks
        a = distance.euclidean(eye[1], eye[5])
        b = distance.euclidean(eye[2], eye[4])
        
        # Horizontal distance
        c = distance.euclidean(eye[0], eye[3])
        
        # Aspect ratio: average vertical distance divided by horizontal distance
        ear = (a + b) / (2.0 * c)
        return ear

    def check_face_movement(self, track_id, current_box, camera_idx):
        """Analyzes micro-movements to distinguish real people from photos"""
        track_info = self.face_tracking.get(track_id)
        if not track_info or 'motion_history' not in track_info:
            # Initialize motion tracking for this face
            if track_id in self.face_tracking:
                self.face_tracking[track_id]['motion_history'] = []
            return 0.5  # Default neutral score
            
        # Get current position
        current_center = ((current_box[0] + current_box[2]) // 2, 
                         (current_box[1] + current_box[3]) // 2)
                         
        # Get previous position
        last_box = track_info['box']
        last_center = ((last_box[0] + last_box[2]) // 2, 
                      (last_box[1] + last_box[3]) // 2)
        
        # Calculate movement distance
        movement = np.sqrt((current_center[0] - last_center[0])**2 + 
                          (current_center[1] - last_center[1])**2)
                          
        # Record this movement
        motion_history = track_info['motion_history']
        motion_history.append(movement)
        
        # Keep recent movement history
        if len(motion_history) > 30:
            motion_history = motion_history[-30:]
        self.face_tracking[track_id]['motion_history'] = motion_history
        
        # Need enough history to analyze patterns
        if len(motion_history) >= 10:
            # Calculate motion variation
            motion_std = np.std(motion_history)
            
            # Interpret patterns:
            if 0.5 < motion_std < 5.0:
                return 0.9  # Natural subtle variations - likely real
            elif motion_std < 0.1:
                return 0.2  # Almost no movement - probably a photo
            elif motion_std > 10.0:
                return 0.7  # Large movements - person moving around
            else:
                return 0.5  # Inconclusive
        else:
            return 0.5  # Not enough data yet

    def check_face_tracking(self, box, camera_idx):
        """Check if a face box matches one we're already tracking"""
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        # Find the best match among tracked faces
        best_tracking_match = None
        best_tracking_iou = 0
        
        for track_id, track_info in self.face_tracking.items():
            # Skip expired or different camera tracks
            if track_info['ttl'] <= 0 or track_info['camera_idx'] != camera_idx:
                continue
                
            last_box = track_info['box']
            last_center_x = (last_box[0] + last_box[2]) // 2
            last_center_y = (last_box[1] + last_box[3]) // 2
            
            # First quick check - are centers close enough?
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            
            if distance < (box_width + last_box[2] - last_box[0]) * 0.3:
                # Then calculate more precise overlap
                x1 = max(box[0], last_box[0])
                y1 = max(box[1], last_box[1])
                x2 = min(box[2], last_box[2])
                y2 = min(box[3], last_box[3])
                
                if x2 > x1 and y2 > y1:
                    overlap = (x2 - x1) * (y2 - y1)
                    box_area = box_width * box_height
                    last_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
                    iou = overlap / (box_area + last_area - overlap)
                    
                    if iou > best_tracking_iou:
                        best_tracking_iou = iou
                        best_tracking_match = track_id
        
        # Return the matched identity if we found one
        if best_tracking_match and best_tracking_iou > 0.3:
            track_info = self.face_tracking[best_tracking_match]
            
            # Only return known identities
            if track_info['identity'] != "Unknown":
                # Update tracking box
                track_info['box'] = box
                track_info['ttl'] = self.tracking_ttl
                return track_info['identity'], track_info['similarity']
        
        return "Unknown", 0.0

    def load_student_data(self):
        try:
            if os.path.exists(self.attendance_file):
                df = pd.read_excel(self.attendance_file)
                self.student_data = {}
                for _, row in df.iterrows():
                    self.student_data[str(row['ID'])] = {'name': row['Name']}
                self.log_status(f"Loaded {len(self.student_data)} student records")
                
                # Update total students count
                total_count = len(self.student_data)
                self.total_students_var.set(str(total_count))
                self.absent_students_var.set(str(total_count))
                
                # Initialize summary variables as well
                self.summary_total_var.set(str(total_count))
                self.summary_absent_var.set(f"{total_count} (100.0%)")
                self.summary_present_var.set("0 (0.0%)")
                self.summary_rate_var.set("0.0%")
                self.summary_unknown_var.set("0")
                self.summary_fake_var.set("0")
                self.summary_detected_var.set("0")
                
                # Initialize the rate bar
                self.root.after(100, self.initialize_rate_bar)
            else:
                self.log_status("Warning: Attendance sheet not found!")
                self.student_data = {}
                self.total_students_var.set("0")
                self.absent_students_var.set("0")
                self.summary_total_var.set("0")
                self.summary_absent_var.set("0 (0.0%)")
                self.summary_present_var.set("0 (0.0%)")
        except Exception as e:
            self.log_status(f"Error loading student data: {str(e)}")
            self.student_data = {}
            self.total_students_var.set("0")
            self.absent_students_var.set("0")
            
    def initialize_rate_bar(self):
        """Initialize the rate bar after window has fully loaded"""
        width = self.rate_canvas.winfo_width()
        if width < 10:
            # If not rendered yet, try again later
            self.root.after(100, self.initialize_rate_bar)
            return
            
        self.rate_canvas.delete("all")
        self.rate_canvas.create_rectangle(0, 0, width, 20, fill='#FF5252', outline='')

    def update_attendance_display(self, person_id, confidence_str):
        item = self.attendance_tree.get_children()
        for i in item:
            if self.attendance_tree.item(i)['values'][0] == person_id:
                values = list(self.attendance_tree.item(i)['values'])
                values[4] = confidence_str
                self.attendance_tree.item(i, values=values)

    def update_camera_status(self):
        for i in range(len(self.camera_vars)):
            is_active = self.camera_vars[i].get()
            self.active_cameras[i] = is_active
            status = "activated" if is_active else "deactivated"
            self.log_status(f"Camera {i+1} {status}")
            if not is_active and i < len(self.video_labels):
                self.video_labels[i].configure(image='', text="Camera Off", background='#f0f0f0')
                if i in self.camera_frames:
                    self.camera_frames[i] = None
            if is_active and (i < len(self.cameras)) and (self.cameras[i] is None or not self.cameras[i].isOpened()):
                self.log_status(f"Attempting to reconnect Camera {i+1}...")
                try:
                    source = self.camera_sources[i]
                    camera = cv2.VideoCapture(source)
                    if camera.isOpened():
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                        if source.endswith(('.mp4', '.avi', '.mov')):
                            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_width = 1280
                            frame_height = 720
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                        self.cameras[i] = camera
                        self.log_status(f"Camera {i+1} reconnected successfully")
                        if i < len(self.camera_threads) and not self.camera_threads[i].is_alive():
                            queue = Queue(maxsize=1)
                            self.frame_queues[i] = queue
                            thread = threading.Thread(target=self.camera_thread, args=(i, queue), daemon=True)
                            thread.start()
                            self.camera_threads[i] = thread
                    else:
                        self.log_status(f"Failed to reconnect Camera {i+1}")
                except Exception as e:
                    self.log_status(f"Error reconnecting Camera {i+1}: {str(e)}")

    def calculate_multi_camera_confidence(self):
        for person_id in self.active_tracking:
            if person_id not in self.face_confidence:
                self.face_confidence[person_id] = 0
            if person_id not in self.confidence_history:
                self.confidence_history[person_id] = []
            cameras_with_detection = []
            camera_confidences = []
            detection_times = {}
            person_detected = False
            detection_weight = 0
            for camera_idx in range(len(self.cameras)):
                if not self.active_cameras[camera_idx]:
                    continue
                camera_key = f"{camera_idx}_{person_id}"
                if camera_key in self.face_detections:
                    detections = sum(self.face_detections[camera_key])
                    total_frames = len(self.face_detections[camera_key])
                    if total_frames > 0:
                        detection_ratio = detections / total_frames
                        camera_confidence = min(100, detection_ratio * 150)
                        consecutive = self.count_consecutive_ones(self.face_detections[camera_key])
                        if consecutive >= 3:
                            camera_confidence = min(100, camera_confidence * 1.2)
                        camera_confidences.append(camera_confidence)
                        cameras_with_detection.append(camera_idx)
                        detection_times[camera_idx] = time.time()
                        person_detected = True
                        detection_weight += consecutive
            if person_detected:
                if len(camera_confidences) > 0:
                    best_confidence = max(camera_confidences)
                    if len(cameras_with_detection) > 1:
                        detection_timestamps = list(detection_times.values())
                        max_time_diff = max(detection_timestamps) - min(detection_timestamps)
                        if max_time_diff < 2.0:
                            is_adjacent = self.are_cameras_adjacent(cameras_with_detection)
                            if is_adjacent:
                                self.log_status(f"Likely duplicate: {person_id} in adjacent cameras {cameras_with_detection}")
                                best_confidence = min(self.max_confidence, best_confidence * 1.05)
                            else:
                                # Non-adjacent cameras showing the same person - could be different people
                                # If confidence is already high, reduce it a bit to be cautious
                                if best_confidence > 75:
                                    best_confidence = best_confidence * 0.95
                
                if len(cameras_with_detection) > 0:
                    # Final confidence calculation
                    confidence_str = f"{best_confidence:.1f}%"
                    
                    # Update confidence history
                    if person_id not in self.confidence_history:
                        self.confidence_history[person_id] = []
                    self.confidence_history[person_id].append(best_confidence)
                    
                    # Keep only the last 10 confidence values
                    if len(self.confidence_history[person_id]) > 10:
                        self.confidence_history[person_id] = self.confidence_history[person_id][-10:]
                    
                    # Update display
                    self.update_attendance_display(person_id, confidence_str)
                    
                    # Update the counters for attendance statistics
                    if self.recognition_active:
                        present_count = sum(1 for p_id in self.active_tracking)
                        absent_count = len(self.student_data) - present_count
                        self.present_students_var.set(str(present_count))
                        self.absent_students_var.set(str(absent_count))
                else:
                    # If no detections for a frame, add a zero confidence value
                    if person_id in self.confidence_history:
                        self.confidence_history[person_id].append(0)
                        if len(self.confidence_history[person_id]) > 10:
                            self.confidence_history[person_id] = self.confidence_history[person_id][-10:]
    
    def count_consecutive_ones(self, arr):
        max_consecutive = 0
        current_consecutive = 0
        for val in arr:
            if val == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive
    
    def are_cameras_adjacent(self, camera_indices):
        """
        Determines if cameras are physically adjacent to help identify duplicate detections.
        In a real deployment, this would use actual camera positions and field of view data.
        """
        if len(camera_indices) <= 1:
            return False
        
        # Simple check based on consecutive indices
        camera_indices = sorted(camera_indices)
        for i in range(len(camera_indices) - 1):
            if camera_indices[i + 1] - camera_indices[i] != 1:
                return False
        return True

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()