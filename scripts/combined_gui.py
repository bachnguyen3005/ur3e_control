#!/usr/bin/env python3
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import rospy
from math import pi
from std_msgs.msg import Float64
import pyrealsense2 as rs
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
# Import the specific modules from each application
# For shape detection GUI
try:
    from shape_color_detection import detect_shapes_and_colors, detect_shapes_and_colors_yolo, detect_shapes_and_colors_yolo_seg
except ImportError:
    print("Warning: shape_color_detection module not found. Shape detection functionality will be limited.")

# For UR3e robot control
try:
    from ur3e_hardware_controller import UR3eRealRobotController
except ImportError:
    print("Warning: ur3e_hardware_controller module not found. Robot control functionality will be limited.")

class CombinedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Combined Robot Control and Shape Detection")
        self.root.geometry("1200x800")
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.robot_tab = ttk.Frame(self.notebook)
        self.combined_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.combined_tab, text="Camera & Shape Detection")
        self.notebook.add(self.robot_tab, text="Robot Control")
        
        # Initialize the combined tab
        self.init_combined_tab()
        
        # Initialize the robot control tab
        self.init_robot_control()
        
        # Connect the tabs
        self.combined_gui.robot_gui = self.robot_gui
        
    def init_combined_tab(self):
        """Initialize the combined camera and shape detection interface"""
        self.combined_gui = CombinedCameraShapeTab(self.combined_tab)

    def init_robot_control(self):
        """Initialize the robot control interface in its tab"""
        self.robot_gui = UR3eGUITab(self.robot_tab)

class CombinedCameraShapeTab:
    def __init__(self, parent):
        self.parent = parent
        
        # Variables for shape detection
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.result_image = None
        self.roi_start_x = None
        self.roi_start_y = None
        self.roi_end_x = None
        self.roi_end_y = None
        self.is_drawing_roi = False
        self.current_target = "all"
        self.robot_square_positions = []
        self.robot_circle_positions = []
        self.yolo_model_path = "/home/dinh/catkin_ws/src/ur3e_control/scripts/object_detection.pt"
        self.custom_yolo_model_path = "/home/dinh/catkin_ws/src/ur3e_control/scripts/segment.pt"
        
        # Variables for camera
        self.pipeline = None
        self.is_capturing = False
        self.last_captured_image = None
        
        # Create main frames
        self.create_frames()
        
        # Create widgets
        self.create_widgets()
        
        # Initialize status
        self.update_status("Ready. Use camera or load an image.")
    
    def create_frames(self):
        # Main layout - split into three panels
        self.top_frame = ttk.Frame(self.parent, padding="5")
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Left panel for camera preview
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera Preview")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Center panel for shape detection image
        self.detection_frame = ttk.LabelFrame(self.main_frame, text="Shape Detection")
        self.detection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for controls
        self.controls_frame = ttk.Frame(self.main_frame, width=200)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Status bar at bottom
        self.status_frame = ttk.Frame(self.parent, padding="5")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas for camera preview
        self.camera_canvas = tk.Canvas(self.camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for shape detection with scrollbars
        self.detection_canvas_frame = ttk.Frame(self.detection_frame)
        self.detection_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.detection_canvas = tk.Canvas(self.detection_canvas_frame, bg="gray", highlightthickness=0)
        self.detection_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars for detection canvas
        self.h_scrollbar = ttk.Scrollbar(self.detection_frame, orient=tk.HORIZONTAL, command=self.detection_canvas.xview)
        self.h_scrollbar.pack(fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.detection_canvas_frame, orient=tk.VERTICAL, command=self.detection_canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detection_canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
    def create_widgets(self):
        # Camera controls in top frame
        camera_controls = ttk.Frame(self.top_frame)
        camera_controls.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        ttk.Label(camera_controls, text="RealSense Camera", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.camera_start_btn = ttk.Button(camera_controls, text="Start Camera", command=self.toggle_camera)
        self.camera_start_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(camera_controls, text="Capture Image", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Load image button in top frame
        image_controls = ttk.Frame(self.top_frame)
        image_controls.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(image_controls, text="Load Image File", command=self.load_image).pack(side=tk.RIGHT, padx=5)
        
        # Controls in right panel
        ttk.Label(self.controls_frame, text="Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Target selection
        ttk.Label(self.controls_frame, text="Target Object:").pack(anchor=tk.W, pady=(10,0))
        self.target_var = tk.StringVar(value="all")
        targets = [
            ("All Objects", "all"),
            ("Red Circle", "red_circle"),
            ("Blue Triangle", "blue_triangle"),
            ("Blue Square", "blue_square"),
            ("Red Square", "red_square"),
            ("Yellow Square", "yellow_square")
        ]
        
        for text, value in targets:
            ttk.Radiobutton(
                self.controls_frame, 
                text=text, 
                value=value, 
                variable=self.target_var,
                command=self.update_target
            ).pack(anchor=tk.W, padx=10)
        
        # ROI controls
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(self.controls_frame, text="Region of Interest (ROI)").pack(anchor=tk.W)
        
        ttk.Button(self.controls_frame, text="Clear ROI", command=self.clear_roi).pack(fill=tk.X, pady=5)
        
        # Detection buttons
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        self.detect_button = ttk.Button(
            self.controls_frame, 
            text="Detect Shapes with OpenCV", 
            command=self.detect_shapes,
            state=tk.DISABLED
        )
        self.detect_button.pack(fill=tk.X, pady=5)
        
        self.detect_yolo_button = ttk.Button(
            self.controls_frame, 
            text="Detect Shapes with YOLO", 
            command=self.detect_shapes_yolo,
            state=tk.DISABLED
        )
        self.detect_yolo_button.pack(fill=tk.X, pady=5)
        
        self.detect_custom_yolo_button = ttk.Button(
            self.controls_frame, 
            text="Detect Shapes with custom YOLO", 
            command=self.detect_shapes_custom_yolo,
            state=tk.DISABLED
        )
        self.detect_custom_yolo_button.pack(fill=tk.X, pady=5)
        
        # Reset view button
        ttk.Button(self.controls_frame, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=5)
        
        # Pick and place button
        self.pick_place_button = ttk.Button(
            self.controls_frame,
            text="Pick and Place Squares",
            command=self.pick_and_place_squares,
            state=tk.DISABLED
        )
        self.pick_place_button.pack(fill=tk.X, pady=10)
        
        # Add after your existing pick_place_button
        self.collision_pick_place_button = ttk.Button(
            self.controls_frame,
            text="Pick and Place with Collision Avoidance",
            command=self.pick_and_place_with_collision_avoidance,
            state=tk.DISABLED
        )
        self.collision_pick_place_button.pack(fill=tk.X, pady=5)
        
        # Instructions
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(self.controls_frame, text="Instructions:").pack(anchor=tk.W)
        instructions = (
            "1. Start camera or load an image\n"
            "2. Capture image from camera\n"
            "3. Draw ROI by dragging mouse\n"
            "4. Press Detect button\n"
            "5. View results"
        )
        ttk.Label(self.controls_frame, text=instructions, wraplength=180).pack(anchor=tk.W, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Canvas event bindings for ROI drawing
        self.detection_canvas.bind("<ButtonPress-1>", self.start_roi)
        self.detection_canvas.bind("<B1-Motion>", self.update_roi)
        self.detection_canvas.bind("<ButtonRelease-1>", self.end_roi)
    
    def toggle_camera(self):
        """Start or stop the camera preview"""
        if self.is_capturing:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start the camera preview"""
        try:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            
            self.is_capturing = True
            self.camera_start_btn.config(text="Stop Camera")
            self.capture_btn.config(state=tk.NORMAL)
            
            # Start preview thread
            self.preview_thread = threading.Thread(target=self.update_preview)
            self.preview_thread.daemon = True
            self.preview_thread.start()
            
            self.update_status("Camera preview started")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}")
            self.update_status(f"Error: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera preview"""
        self.is_capturing = False
        
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            
        self.camera_start_btn.config(text="Start Camera")
        self.capture_btn.config(state=tk.DISABLED)
        self.update_status("Camera stopped")
    
    def update_preview(self):
        """Continuously update the camera preview"""
        try:
            last_update_time = 0
            update_interval = 1/20  # Limit updates to 20 fps for smoother UI
            
            while self.is_capturing:
                current_time = time.time()
                
                # Limit update frequency
                if current_time - last_update_time < update_interval:
                    time.sleep(0.005)  # Small sleep to reduce CPU usage
                    continue
                    
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Schedule UI update on main thread
                self.parent.after_idle(lambda img=color_image: self.show_camera_image(img))
                
                last_update_time = current_time
                time.sleep(0.001)  # Small sleep to ensure other threads get CPU time
                
        except Exception as e:
            # Schedule UI updates from the thread
            self.parent.after(0, lambda: self.update_status(f"Preview error: {str(e)}"))
            self.parent.after(0, self.stop_camera)
    
    def show_camera_image(self, image):
        """Display an image on the camera canvas"""
        try:
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit the canvas
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            
            # Ensure we have valid dimensions
            if canvas_width > 1 and canvas_height > 1:
                # Calculate aspect ratio preserving scale
                h, w = image_rgb.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                
                if scale < 1:  # Only scale down, not up
                    new_size = (int(w * scale), int(h * scale))
                    image_rgb = cv2.resize(image_rgb, new_size)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(image_rgb)
            self.camera_image = ImageTk.PhotoImage(pil_image)  # Keep a reference
            
            # Center the image on canvas
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            img_width, img_height = pil_image.size
            x = max(0, (canvas_width - img_width) // 2)
            y = max(0, (canvas_height - img_height) // 2)
            
            # Clear canvas and display image
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(x, y, anchor=tk.NW, image=self.camera_image)
            
        except Exception as e:
            self.parent.after(0, lambda: self.update_status(f"Display error: {str(e)}"))
    
    def capture_image(self):
        """Capture the current camera frame and load it into the detection canvas"""
        if not self.is_capturing or not self.pipeline:
            self.update_status("Camera is not running")
            return
        
        try:
            # Get the current frame
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                self.update_status("Failed to get color frame")
                return
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Store as original image for processing
            self.original_image = color_image.copy()
            self.display_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Clear any existing ROI
            self.clear_roi()
            
            # Show on the detection canvas
            self.show_detection_image(self.display_image)
            
            # Enable detection buttons
            self.detect_button.config(state=tk.NORMAL)
            self.detect_yolo_button.config(state=tk.NORMAL)
            self.detect_custom_yolo_button.config(state=tk.NORMAL)
            
            self.update_status("Image captured and loaded for detection")
            
        except Exception as e:
            self.update_status(f"Capture error: {str(e)}")
    
    def load_image(self):
        """Open a file dialog to select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            try:
                # Load the image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    self.update_status(f"Error: Could not load image from {file_path}")
                    return
                
                # Convert to RGB for display
                self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Clear any existing ROI
                self.clear_roi()
                
                # Display the image
                self.show_detection_image(self.display_image)
                
                self.update_status(f"Loaded image: {os.path.basename(file_path)}")
                self.detect_button.config(state=tk.NORMAL)
                self.detect_yolo_button.config(state=tk.NORMAL)
                self.detect_custom_yolo_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")
    
    def show_detection_image(self, image):
        """Display an image on the detection canvas"""
        # Convert the image to PIL format
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        
        # Convert to PhotoImage
        self.detection_tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.detection_canvas.delete("all")
        self.detection_canvas.config(scrollregion=(0, 0, w, h))
        self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=self.detection_tk_image)
    
    def start_roi(self, event):
        """Start ROI selection on mouse click"""
        if self.original_image is None:
            return
        
        # Get the canvas coordinates
        x = self.detection_canvas.canvasx(event.x)
        y = self.detection_canvas.canvasy(event.y)
        
        self.roi_start_x = int(x)
        self.roi_start_y = int(y)
        self.is_drawing_roi = True
        
        # Clear previous ROI
        self.detection_canvas.delete("roi")
        
    def update_roi(self, event):
        """Update ROI rectangle as mouse moves"""
        if not self.is_drawing_roi:
            return
        
        # Get the canvas coordinates
        x = self.detection_canvas.canvasx(event.x)
        y = self.detection_canvas.canvasy(event.y)
        
        self.roi_end_x = int(x)
        self.roi_end_y = int(y)
        
        # Clear and redraw ROI
        self.detection_canvas.delete("roi")
        self.detection_canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, 
            self.roi_end_x, self.roi_end_y, 
            outline="yellow", width=2, tags="roi"
        )
        
    def end_roi(self, event):
        """Finalize ROI selection on mouse release"""
        if not self.is_drawing_roi:
            return
        
        self.is_drawing_roi = False
        
        # Get the canvas coordinates
        x = self.detection_canvas.canvasx(event.x)
        y = self.detection_canvas.canvasy(event.y)
        
        self.roi_end_x = int(x)
        self.roi_end_y = int(y)
        
        # Ensure start coordinates are smaller than end coordinates
        if self.roi_start_x > self.roi_end_x:
            self.roi_start_x, self.roi_end_x = self.roi_end_x, self.roi_start_x
        
        if self.roi_start_y > self.roi_end_y:
            self.roi_start_y, self.roi_end_y = self.roi_end_y, self.roi_start_y
        
        # Redraw final ROI
        self.detection_canvas.delete("roi")
        self.detection_canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, 
            self.roi_end_x, self.roi_end_y, 
            outline="yellow", width=2, tags="roi"
        )
        
        roi_width = self.roi_end_x - self.roi_start_x
        roi_height = self.roi_end_y - self.roi_start_y
        self.update_status(f"ROI selected: ({self.roi_start_x}, {self.roi_start_y}) to ({self.roi_end_x}, {self.roi_end_y}), size: {roi_width}x{roi_height}")
    
    def clear_roi(self):
        """Clear the selected ROI"""
        self.detection_canvas.delete("roi")
        self.roi_start_x = None
        self.roi_start_y = None
        self.roi_end_x = None
        self.roi_end_y = None
        
        if self.display_image is not None:
            self.show_detection_image(self.display_image)
        
        self.update_status("ROI cleared")
    
    def detect_shapes(self):
        """Run shape detection on the selected ROI or the entire image"""
        if self.original_image is None:
            self.update_status("Please load or capture an image first")
            return
        
        try:
            # Check if the shape detection module is available
            if 'detect_shapes_and_colors' not in globals():
                self.update_status("Shape detection module not available")
                messagebox.showwarning("Module Not Available", 
                                      "The shape_color_detection module is not available. This is a mock interface.")
                return
                
            # Get the target object
            target = self.target_var.get()
            
            # Create a copy of the original image
            image_to_process = self.original_image.copy()
            
            # If ROI is selected, crop the image
            if (self.roi_start_x is not None and self.roi_end_x is not None and 
                self.roi_start_y is not None and self.roi_end_y is not None):
                
                # Ensure ROI is within image boundaries
                h, w = image_to_process.shape[:2]
                start_x = max(0, min(self.roi_start_x, w-1))
                start_y = max(0, min(self.roi_start_y, h-1))
                end_x = max(0, min(self.roi_end_x, w-1))
                end_y = max(0, min(self.roi_end_y, h-1))
                
                # Crop the image
                cropped = image_to_process[start_y:end_y, start_x:end_x]
                
                # Make sure we have a valid crop
                if cropped.size == 0:
                    self.update_status("Invalid ROI: zero size after crop")
                    return
                
                # Process the cropped image with ROI offset for coordinate calculation
                cropped_result, self.detected_squares = detect_shapes_and_colors(cropped, target, roi_offset=(start_x, start_y))
                
                # Create a full-sized result image
                self.result_image = image_to_process.copy()
                
                # Place the processed cropped region back into the full image
                self.result_image[start_y:end_y, start_x:end_x] = cropped_result
                
                # Draw a rectangle around the ROI in the result image
                cv2.rectangle(self.result_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
            else:
                # Process the entire image - no ROI offset needed
                self.result_image, self.detected_squares = detect_shapes_and_colors(image_to_process, target)
            
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.show_detection_image(display_result)
            
            self.update_status(f"Detection completed for target: {target}")
            
            # Map squares to robot coordinates
            if self.detected_squares:
                self.map_squares_to_robot_frame()
                self.pick_place_button.config(state=tk.NORMAL)
                self.collision_pick_place_button.config(state=tk.NORMAL)
                self.update_status(f"Found {len(self.detected_squares)} squares. Ready for pick and place.")
               
            else:
                self.pick_place_button.config(state=tk.DISABLED)
                self.collision_pick_place_button.config(state=tk.DISABLED)
                self.update_status("No squares detected.")
            
        except Exception as e:
            self.update_status(f"Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def detect_shapes_yolo(self):
        """Run YOLO shape detection on the selected ROI or the entire image"""
        if self.original_image is None:
            self.update_status("Please load or capture an image first")
            return
        try:
            # Get the target object
            target = self.target_var.get()
            
            # Check if model file exists
            if not os.path.isfile(self.yolo_model_path):
                self.update_status(f"Error: YOLO model not found at {self.yolo_model_path}")
                return
            
            # Create a copy of the original image
            image_to_process = self.original_image.copy()
            
            # If ROI is selected, crop the image
            if (self.roi_start_x is not None and self.roi_end_x is not None and 
                self.roi_start_y is not None and self.roi_end_y is not None):
                
                # Ensure ROI is within image boundaries
                h, w = image_to_process.shape[:2]
                start_x = max(0, min(self.roi_start_x, w-1))
                start_y = max(0, min(self.roi_start_y, h-1))
                end_x = max(0, min(self.roi_end_x, w-1))
                end_y = max(0, min(self.roi_end_y, h-1))
                
                # Crop the image
                cropped = image_to_process[start_y:end_y, start_x:end_x]
                
                # Make sure we have a valid crop
                if cropped.size == 0:
                    self.update_status("Invalid ROI: zero size after crop")
                    return
                
                # Process the cropped image using YOLO
                self.update_status("Running YOLO detection on ROI...")
                cropped_result, self.detected_squares, self.detected_circle = detect_shapes_and_colors_yolo(
                    cropped, 
                    target, 
                    model_path=self.yolo_model_path
                )
                
                # Create a full-sized result image
                self.result_image = image_to_process.copy()
                
                # Place the processed cropped region back into the full image
                self.result_image[start_y:end_y, start_x:end_x] = cropped_result
                
                # Draw a rectangle around the ROI in the result image
                cv2.rectangle(self.result_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
                # Add ROI reference text to the full image
                roi_width = end_x - start_x
                roi_height = end_y - start_y
                full_width, full_height = image_to_process.shape[1], image_to_process.shape[0]
                
                roi_text = f"ROI: ({start_x}, {start_y}) to ({end_x}, {end_y}), size: {roi_width}x{roi_height}"
                cv2.putText(self.result_image, roi_text, (10, full_height - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add YOLO model info
                model_name = os.path.basename(self.yolo_model_path)
                cv2.putText(self.result_image, f"YOLO Model: {model_name}", (10, full_height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            else:
                # Process the entire image with YOLO
                self.update_status("Running YOLO detection on full image...")
                self.result_image, self.detected_squares, self.detected_circle = detect_shapes_and_colors_yolo(
                    image_to_process, 
                    target, 
                    model_path=self.yolo_model_path
                )
                print("DEBUG: ", self.detected_squares)
                # Add YOLO model info
                h, w = self.result_image.shape[:2]
                model_name = os.path.basename(self.yolo_model_path)
                cv2.putText(self.result_image, f"YOLO Model: {model_name}", (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.show_detection_image(display_result)
            
            # Report detected squares
            if self.detected_squares:
                square_info = ", ".join([f"{color} at ({x},{y})" for x, y, color in self.detected_squares])
                self.update_status(f"YOLO detection completed. Found squares: {square_info}")
                self.map_squares_to_robot_frame()
                self.pick_place_button.config(state=tk.NORMAL)
                self.collision_pick_place_button.config(state=tk.NORMAL)
            else:
                self.pick_place_button.config(state=tk.DISABLED)
                self.collision_pick_place_button.config(state=tk.DISABLED)
                self.update_status(f"YOLO detection completed for target: {target}. No squares found.")
            
            if self.detected_circle: 
                circle_info = ", ".join([f"{color} at ({x},{y})" for x, y, color in self.detected_circle])
                self.update_status(f"YOLO detection completed. Found circle: {circle_info}")
                self.map_circle_to_robot_frame()
        except Exception as e:
            self.update_status(f"Error during YOLO detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def detect_shapes_custom_yolo(self):
        """Run YOLO shape detection on the selected ROI or the entire image"""
        if self.original_image is None:
            self.update_status("Please load or capture an image first")
            return
        try:
            # Get the target object
            target = self.target_var.get()
            
            # Check if model file exists
            if not os.path.isfile(self.custom_yolo_model_path):
                self.update_status(f"Error: YOLO model not found at {self.custom_yolo_model_path}")
                return
            
            # Create a copy of the original image
            image_to_process = self.original_image.copy()
            
            # If ROI is selected, crop the image
            if (self.roi_start_x is not None and self.roi_end_x is not None and 
                self.roi_start_y is not None and self.roi_end_y is not None):
                
                # Ensure ROI is within image boundaries
                h, w = image_to_process.shape[:2]
                start_x = max(0, min(self.roi_start_x, w-1))
                start_y = max(0, min(self.roi_start_y, h-1))
                end_x = max(0, min(self.roi_end_x, w-1))
                end_y = max(0, min(self.roi_end_y, h-1))
                
                # Crop the image
                cropped = image_to_process[start_y:end_y, start_x:end_x]
                
                # Make sure we have a valid crop
                if cropped.size == 0:
                    self.update_status("Invalid ROI: zero size after crop")
                    return
                
                # Process the cropped image using YOLO
                self.update_status("Running custom YOLO detection on ROI...")
                cropped_result, self.detected_squares, self.detected_circle = detect_shapes_and_colors_yolo_seg(
                    cropped, 
                    target, 
                    model_path=self.custom_yolo_model_path
                )
                
                # Create a full-sized result image
                self.result_image = image_to_process.copy()
                
                # Place the processed cropped region back into the full image
                self.result_image[start_y:end_y, start_x:end_x] = cropped_result
                
                # Draw a rectangle around the ROI in the result image
                cv2.rectangle(self.result_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
                # Add ROI reference text to the full image
                roi_width = end_x - start_x
                roi_height = end_y - start_y
                full_width, full_height = image_to_process.shape[1], image_to_process.shape[0]
                
                roi_text = f"ROI: ({start_x}, {start_y}) to ({end_x}, {end_y}), size: {roi_width}x{roi_height}"
                cv2.putText(self.result_image, roi_text, (10, full_height - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add YOLO model info
                model_name = os.path.basename(self.yolo_model_path)
                cv2.putText(self.result_image, f"YOLO Model: {model_name}", (10, full_height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            else:
                # Process the entire image with YOLO
                self.update_status("Running custom YOLO detection on full image...")
                self.result_image, self.detected_squares, self.detected_circle = detect_shapes_and_colors_yolo_seg(
                    image_to_process, 
                    target, 
                    model_path=self.custom_yolo_model_path
                )
                
                # Add YOLO model info
                h, w = self.result_image.shape[:2]
                model_name = os.path.basename(self.custom_yolo_model_path)
                cv2.putText(self.result_image, f"YOLO Model: {model_name}", (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.show_detection_image(display_result)
            
            # Report detected squares
            if self.detected_squares:
                square_info = ", ".join([f"{color} at ({x},{y}), roation is {rotation}" for x, y, color, rotation in self.detected_squares])
                self.update_status(f"YOLO detection completed. Found squares: {square_info}")
                self.map_squares_to_robot_frame()
                self.pick_place_button.config(state=tk.NORMAL)
                self.collision_pick_place_button.config(state=tk.NORMAL)
            else:
                self.pick_place_button.config(state=tk.DISABLED)
                self.collision_pick_place_button.config(state=tk.DISABLED)
                self.update_status(f"YOLO detection completed for target: {target}. No squares found.")
            
            if self.detected_circle: 
                circle_info = ", ".join([f"{color} at ({x},{y})" for x, y, color in self.detected_circle])
                self.update_status(f"YOLO detection completed. Found circle: {circle_info}")
                self.map_circle_to_robot_frame()
        except Exception as e:
            self.update_status(f"Error during YOLO detection: {str(e)}")
            import traceback
            traceback.print_exc()            

    def update_target(self):
        """Update the current target object"""
        self.current_target = self.target_var.get()
        self.update_status(f"Target changed to: {self.current_target}")
    
    def reset_view(self):
        """Reset to the original image view"""
        if self.original_image is not None:
            # Convert to RGB for display
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Show original image
            self.show_detection_image(self.display_image)
            
            # Redraw ROI if it exists
            if (self.roi_start_x is not None and self.roi_end_x is not None and 
                self.roi_start_y is not None and self.roi_end_y is not None):
                
                self.detection_canvas.create_rectangle(
                    self.roi_start_x, self.roi_start_y, 
                    self.roi_end_x, self.roi_end_y, 
                    outline="yellow", width=2, tags="roi"
                )
            
            self.update_status("View reset to original image")
    
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_var.set(message)
        print(message)
        
    def map_squares_to_robot_frame(self):
        """Map pixel coordinates of squares to robot frame coordinates"""
        from Image_to_world_mapping import map_pixel_to_robot_frame
        
        # Camera intrinsic parameters
        camera_intrinsics = {
            'height': 480,
            'width': 640,
            'distortion_model': 'plumb_bob',
            'D': [0.06117127, 0.1186219, -0.00319266, -0.00094209, -0.75616137],
            'K': [596.68849646, 0.0, 317.08346319,
                0.0, 596.0051831, 247.34662529,
                0.0, 0.0, 1.0],
            'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'P': [596.68849646, 0.0, 317.0835119, 0.0,
                0.0, 596.0051831, 247.34658383, 0.0,
                0.0, 0.0, 1.0, 0.0]
        }
        
        marker_size_mm = 36  # Adjust as needed
        
        # Map each square's position
        self.robot_square_positions = []
        for square in self.detected_squares:
            pixel_x, pixel_y, color  = square
            try:
                result = map_pixel_to_robot_frame(
                    "/home/dinh/catkin_ws/src/ur3e_control/calibration_images.jpg", pixel_x, pixel_y, 
                    camera_intrinsics, marker_size_mm
                )
                robot_x, robot_y = result[0]
                self.robot_square_positions.append((robot_x, robot_y, color))
                self.update_status(f"Mapped square at ({pixel_x},{pixel_y}) to robot coordinates ({robot_x:.2f},{robot_y:.2f})")
            except Exception as e:
                self.update_status(f"Error mapping square: {str(e)}")
    
    def map_circle_to_robot_frame(self):
        """Map pixel coordinates of squares to robot frame coordinates"""
        from Image_to_world_mapping import map_pixel_to_robot_frame
        
        # Camera intrinsic parameters
        camera_intrinsics = {
            'height': 480,
            'width': 640,
            'distortion_model': 'plumb_bob',
            'D': [0.06117127, 0.1186219, -0.00319266, -0.00094209, -0.75616137],
            'K': [596.68849646, 0.0, 317.08346319,
                0.0, 596.0051831, 247.34662529,
                0.0, 0.0, 1.0],
            'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'P': [596.68849646, 0.0, 317.0835119, 0.0,
                0.0, 596.0051831, 247.34658383, 0.0,
                0.0, 0.0, 1.0, 0.0]
        }
        
        marker_size_mm = 36  # Adjust as needed
        
        # Map each square's position
        self.robot_circle_positions = []
        for circle in self.detected_circle:
            pixel_x, pixel_y, color = circle
            try:
                result = map_pixel_to_robot_frame(
                    "/home/dinh/catkin_ws/src/ur3e_control/calibration_images.jpg", pixel_x, pixel_y, 
                    camera_intrinsics, marker_size_mm
                )
                robot_x, robot_y = result[0]
                self.robot_circle_positions.append((robot_x, robot_y, color))
                print("DEGBUG: Circle position with robot: ", [robot_x, robot_y])
                self.update_status(f"Mapped circle at ({pixel_x},{pixel_y}) to robot coordinates ({robot_x:.2f},{robot_y:.2f})")
            except Exception as e:
                self.update_status(f"Error mapping circle: {str(e)}")    
    
    def pick_and_place_squares(self):
        """Execute pick and place sequence for all detected squares"""
        if not hasattr(self, 'robot_square_positions') or not self.robot_square_positions:
            self.update_status("No squares mapped to robot coordinates")
            return
        
        # Access the robot_gui through the reference set in the CombinedGUI.__init__ method
        if not hasattr(self, 'robot_gui') or self.robot_gui is None:
            self.update_status("Robot control not available - direct reference not set")
            return
        
        robot_gui = self.robot_gui
        self.update_status("Found robot control interface. Proceeding with pick and place.")
        
        # Execute sequence for each square
        for i, (robot_x, robot_y, color) in enumerate(self.robot_square_positions):
            self.update_status(f"Processing {color} square at ({robot_x:.2f}, {robot_y:.2f})...")
            
            # Generate pickup pose
            pickup_pose = self.calculate_pickup_pose(robot_x, robot_y)
            
            # Generate dropoff pose based on color and index
            dropoff_pose = self.determine_dropoff_location(color, i)
            
            # Execute the pick and place sequence
            success = self.execute_pick_and_place(robot_gui, pickup_pose, dropoff_pose)
            
            if success:
                self.update_status(f"Successfully picked and placed {color} square")
            else:
                self.update_status(f"Failed to pick and place {color} square")
                
    def calculate_pickup_pose(self, robot_x, robot_y):
        """Calculate robot joint positions for picking up an object"""
        import numpy as np
        from math import pi
        import roboticstoolbox as rtb
        from spatialmath.base import transl, rpy2tr
        from ikcon import ikcon
        
        # Create robot model
        ur3 = rtb.models.UR3()
        
        # Calculate end-effector transform for the pickup position
        # Note: Coordinate conversion and offset adjustments may be needed
        Tep = transl(-(robot_x+15)*0.001, -(robot_y+10)*0.001, 0.257+0.015) @ rpy2tr(0, pi/2, pi/2)
        
        # Preferred joint configuration for IK solution
        q_prefer = np.deg2rad(np.array([61.26, -81.48, -92.51, -91.86, 85.49, 6.96]))
        
        # Solve inverse kinematics
        sol, err, flag, out = ikcon(ur3, Tep, q0=q_prefer)
        
        # Return joint angles in degrees
        return np.rad2deg(sol)

    def determine_dropoff_location(self, color, index):
        """Determine dropoff location based on object color and index"""
        # Predefined dropoff positions - note these are angles in degrees, not a preset name
        dropoff_positions = {
            "red": [-162.37, -82.71, -111.48, -75.72, 90.03, -72.62], #-288.71, 46.79
            "blue": [-145.41, -89.24, -105.50, -75.17, 90.100, -55.42], #311.78, -55.44
            "yellow": [-131.68, -96.94, -97.32, -75.65, 90.06, 318.31-360]#-311.84, -152.49
        }
        
        # Get position based on color
        if color in dropoff_positions:
            return dropoff_positions[color]
        else:
            # Default position if color not recognized
            return dropoff_positions["red"]

    def execute_pick_and_place(self, robot_gui, pickup_pose, dropoff_pose):
        """Execute the pick and place sequence using the robot control with intermediate poses"""
        try:
            import time  # Import time module for delays
        
            # Define delay times in seconds
            move_delay = 1.0      # Delay after robot movement
            gripper_delay = 1.0   # Delay after gripper operation
            
            # Define an intermediate/safe pose (slightly above the workspace)
            # These values are in degrees - adjust as needed for your robot setup
            intermediate_pose = [59.25, -89.96, 67.24, -67.32, -89.04, -30]  # Example safe position above the workspace
            capture_image_pose = [-62.71, -88.97, -31.19, -149.76, 89.85, 27.15]
            
            self.update_status("Opening gripper...")
            robot_gui.set_gripper_width("0.1")
            robot_gui.send_gripper_command()
            time.sleep(gripper_delay)
            
            # 2. Move to a pre-pickup pose (slightly above the pickup position)
            self.update_status("Moving to pre-pickup pose...")
            pre_pickup_pose = pickup_pose.copy()  # Create a copy to modify
            pre_pickup_pose[2] += 20  # Adjust the 3rd joint to be higher (adjust as needed)
            robot_gui.set_joint_values(pre_pickup_pose)
            robot_gui.move_robot()
            
            # 3. Move to the actual pickup position
            self.update_status("Moving to pickup position...")
            robot_gui.set_joint_values(pickup_pose)
            robot_gui.move_robot()
            
            # 4. Close gripper to grasp object
            self.update_status("Closing gripper...")
            robot_gui.set_gripper_width("0.056")
            robot_gui.send_gripper_command()
            time.sleep(gripper_delay)  # Wait for movement to complete
            
            # 5. Lift object back to pre-pickup (post-pickup) position
            self.update_status("Lifting object...")
            robot_gui.set_joint_values(pre_pickup_pose)
            robot_gui.move_robot()
            
            # # 7. Move to pre-dropoff position
            self.update_status("Moving to pre-dropoff position...")
            pre_dropoff_pose = dropoff_pose.copy()
            pre_dropoff_pose[2] += 20  # Adjust height for pre-dropoff
            robot_gui.set_joint_values(pre_dropoff_pose)
            robot_gui.move_robot()
            time.sleep(move_delay)  # Wait for movement to complete
            
            # 8. Move to final dropoff position
            self.update_status("Moving to dropoff position...")
            robot_gui.set_joint_values(dropoff_pose)
            robot_gui.move_robot()
            
            # 9. Open gripper to release object
            self.update_status("Opening gripper...")
            robot_gui.set_gripper_width("0.1")
            robot_gui.send_gripper_command()
            time.sleep(gripper_delay)
            
            # # 10. Move back to pre-dropoff position
            # self.update_status("Moving to post-dropoff position...")
            # robot_gui.set_joint_values(pre_dropoff_pose)
            # robot_gui.move_robot()
            
            # 11. Return to intermediate pose
            self.update_status("Returning to intermediate pose...")
            robot_gui.set_joint_values(capture_image_pose)
            robot_gui.move_robot()
            
            self.update_status("Pick and place sequence completed successfully!")
            return True
        except Exception as e:
            self.update_status(f"Error during pick and place: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def execute_pick_phase(self, robot_gui, pickup_pose):
        """Execute the pick and place sequence using the robot control with intermediate poses"""
        try:
            import time  # Import time module for delays
        
            # Define delay times in seconds
            move_delay = 1.0      # Delay after robot movement
            gripper_delay = 1.0   # Delay after gripper operation
            
            # Define an intermediate/safe pose (slightly above the workspace)
            # These values are in degrees - adjust as needed for your robot setup
            intermediate_pose = [59.25, -89.96, 67.24, -67.32, -89.04, -30]  # Example safe position above the workspace
            capture_image_pose = [-62.71, -88.97, -31.19, -149.76, 89.85, 27.15]
            
            self.update_status("Opening gripper...")
            robot_gui.set_gripper_width("0.1")
            robot_gui.send_gripper_command()
            time.sleep(gripper_delay)
            
            # 2. Move to a pre-pickup pose (slightly above the pickup position)
            self.update_status("Moving to pre-pickup pose...")
            pre_pickup_pose = pickup_pose.copy()  # Create a copy to modify
            pre_pickup_pose[2] += 20  # Adjust the 3rd joint to be higher (adjust as needed)
            robot_gui.set_joint_values(pre_pickup_pose)
            robot_gui.move_robot()
            
            # 3. Move to the actual pickup position
            self.update_status("Moving to pickup position...")
            robot_gui.set_joint_values(pickup_pose)
            robot_gui.move_robot()
            
            # 4. Close gripper to grasp object
            self.update_status("Closing gripper...")
            robot_gui.set_gripper_width("0.056")
            robot_gui.send_gripper_command()
            time.sleep(gripper_delay)  # Wait for movement to complete
            
            # 5. Lift object back to pre-pickup (post-pickup) position
            self.update_status("Lifting object...")
            robot_gui.set_joint_values(pre_pickup_pose)
            robot_gui.move_robot()
            
            self.update_status("Pick phase was successful!")
            return True
        except Exception as e:
            self.update_status(f"Error during pick and place: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_collision_avoidance_path(self, square_index=0, save_csv=True): 
        try: 
            # Check if we have squares mapped
            if not hasattr(self, 'robot_square_positions') or not self.robot_square_positions:
                self.update_status("No squares mapped to robot coordinates")
                return False, None
            
            # === Get Input Parameters ===
            # Get square position (start)
            if square_index >= len(self.robot_square_positions):
                self.update_status(f"Error: Invalid square index {square_index}")
                return False, None
            
            square_x, square_y, square_color = self.robot_square_positions[square_index]
            
            # === Define Hardcoded Drop-off Locations by Color ===
            # These are X, Y, Z coordinates in the robot's workspace
            dropoff_locations = {
                "red":    [-0.28871, 0.04679, 0.04],  # Hardcoded position for red cubes -288.71, 46.79
                "blue":   [-0.31178, -0.05544, 0.04],  # Hardcoded position for blue cubes 311.78, -55.44
                "yellow": [-0.31184, -0.15249, 0.04],  # Hardcoded position for yellow cubes -311.84, -152.49
                # Add more colors and positions as needed
            }
            
            # Get the drop-off location based on color
            if square_color.lower() in dropoff_locations:
                dropoff_x, dropoff_y, dropoff_z = dropoff_locations[square_color.lower()]
                self.update_status(f"Using hardcoded drop-off location for {square_color} cube")
            
            # === Parameters ===
            # Starting position (cube to pick up)
            P0 = np.array([square_x*0.001, square_y*0.001, 0.120])  # [x, y, z]
            
            # Target position (drop-off location)
            P2 = np.array([dropoff_x, dropoff_y, dropoff_z])  # [x, y, z]
            
            # Default quaternion orientation
            Q0 = np.array([0, 0, 0, 1])  # [qx, qy, qz, qw]
            
            # Safety parameters
            safeMargin = 0.1  # extra clearance (m)
            projectedCylinderRadius = 0.1  # cylinder radius (m)
            R = projectedCylinderRadius + safeMargin
            zConstant = P0[2]  # keep Z constant throughout the path
            
            # Robot base (always an obstacle)
            base_center = np.array([0.0, 0.0])  # robot-base at origin (XY only)
            base_R = 0.25  # 175 mm radius → 350 mm Ø
            
            # obstacles = []
            
            # # Add detected circles as obstacles
            # if hasattr(self, 'robot_circle_positions') and self.robot_circle_positions:
            #     for i, (cx, cy, color) in enumerate(self.robot_circle_positions):
            #         c_xy = np.array([cx, cy])
            #         print("Cylinder position:  ", c_xy)
            #         obstacles.append((c_xy, R, f'cylinder {i} ({color})'))
            
            # # Always add robot base as an obstacle
            # obstacles.append((base_center, base_R, 'robot base'))     
                  
            # === Helper functions ===
            def wrap_to_pi(angle):
                return (angle + math.pi) % (2 * math.pi) - math.pi

            def check_collision(p0_xy, p1_xy, c_xy, radius):
                """True if the segment p0→p1 comes within 'radius' of c_xy."""
                v = p1_xy - p0_xy
                w = c_xy - p0_xy
                t = np.clip(np.dot(w, v) / np.dot(v, v), 0.0, 1.0)
                closest = p0_xy + t*v
                return np.linalg.norm(c_xy - closest) < radius

            def compute_tangent_path(p0_xy, p2_xy, c_xy, radius, n_intermediate=1):
                """
                Return a smooth detour [p0, T1, mid…, T2, p2] around circle at c_xy.
                Ensures T1 is the tangent closest to p0 to avoid any 'bounce'.
                """
                # angles from center to endpoints
                θ0 = math.atan2(p0_xy[1]-c_xy[1], p0_xy[0]-c_xy[0])
                θ2 = math.atan2(p2_xy[1]-c_xy[1], p2_xy[0]-c_xy[0])
                d0, d2 = np.linalg.norm(p0_xy - c_xy), np.linalg.norm(p2_xy - c_xy)
                α0 = math.acos(radius / d0)
                α2 = math.acos(radius / d2)

                cand0 = [θ0 + α0, θ0 - α0]
                cand2 = [θ2 + α2, θ2 - α2]

                # pick the pair whose absolute angular sweep is smallest
                best = float('inf')
                sel0 = sel2 = None
                for a0 in cand0:
                    for a2 in cand2:
                        sweep = wrap_to_pi(a2 - a0)
                        if abs(sweep) < best:
                            best, sel0, sel2 = abs(sweep), a0, a2

                # compute the two tangent points
                T1 = c_xy + radius * np.array([math.cos(sel0), math.sin(sel0)])
                T2 = c_xy + radius * np.array([math.cos(sel2), math.sin(sel2)])

                # if T1 is farther from p0 than T2, swap them (so we always go to the closer one first)
                if np.linalg.norm(p0_xy - T1) > np.linalg.norm(p0_xy - T2):
                    T1, T2 = T2, T1
                    sel0, sel2 = sel2, sel0

                # now sample a few points along the shorter arc from sel0→sel2
                Δθ = wrap_to_pi(sel2 - sel0)
                ts = np.linspace(0, 1, n_intermediate+2)  # includes endpoints
                arc_pts = [c_xy + radius * np.array([math.cos(sel0 + t*Δθ),
                                                math.sin(sel0 + t*Δθ)])
                        for t in ts]

                # build [p0, arc_pts..., p2]
                return np.vstack([p0_xy] + arc_pts + [p2_xy])

            def circles_intersect(c1, r1, c2, r2):
                return np.linalg.norm(c1 - c2) < (r1 + r2)

            def projection_t(p0, p1, c):
                v = p1 - p0
                w = c - p0
                return np.clip(np.dot(w, v) / np.dot(v, v), 0.0, 1.0)

            def compute_detour_sequence(p0, p2, obstacles):
                hits = []
                for c_xy, radius, name in obstacles:
                    if check_collision(p0, p2, c_xy, radius):
                        hits.append((projection_t(p0, p2, c_xy), c_xy, radius, name))
                if not hits:
                    return np.vstack([p0, p2])

                hits.sort(key=lambda x: x[0])
                waypoints = [p0]
                current = p0

                for _, c_xy, radius, name in hits:
                    self.update_status(f"Collision with {name} → detouring around it")
                    detour = compute_tangent_path(current, p2, c_xy, radius, n_intermediate=1)
                    # detour = [current, T1, mid..., T2, p2]; keep only T1, mid…, T2
                    waypoints.extend(detour[1:-1])
                    current = detour[-2]  # last arc‐point before p2

                waypoints.append(p2)
                return np.vstack(waypoints)
            
            # === Main flow ===
            P0_xy, P2_xy = P0[:2], P2[:2]
            
            # Build the obstacles list
            obstacles = []
            
            # Add detected circles as obstacles
            if hasattr(self, 'robot_circle_positions') and self.robot_circle_positions:
                for i, (cx, cy, color) in enumerate(self.robot_circle_positions):
                    c_xy = np.array([cx, cy])
                    print("Cylinder position:  ", c_xy)
                    obstacles.append((c_xy, R, f'cylinder {i} ({color})'))
                    
                    # Check if circles overlap with base
                    if circles_intersect(c_xy, R, base_center, base_R):
                        self.update_status(f"⚠️ Warning: {color} circle and robot base no-go zones overlap!")
            
            # Always add robot base as an obstacle
            obstacles.append((base_center, base_R, 'robot base'))
            
            
            # Compute the collision-free path
            self.update_status(f"Computing collision-free path for {square_color} square...")
            pts_xy = compute_detour_sequence(P0_xy, P2_xy, obstacles)
            print("DEBUG pts_xy: ", pts_xy)
            
            # === Save waypoints ===
            N = pts_xy.shape[0]
            xyz = np.hstack([pts_xy, np.full((N, 1), zConstant)])
            quat = np.tile(Q0, (N, 1))
            df = pd.DataFrame(np.hstack([xyz, quat]),
                            columns=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            
            if save_csv:
                # Get scripts directory path
                scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
                os.makedirs(scripts_dir, exist_ok=True)
                
                csv_path = os.path.join(scripts_dir, "waypointsMatrix.csv")
                df.to_csv(csv_path, index=False)
                self.update_status(f"Saved waypoints to {csv_path}")
            
            # === Visualize the path if requested ===
            self.update_status(f"Generated collision-free path with {N} waypoints")
            
            # Return success and the DataFrame
            return True, df   
    
        except Exception as e:
            self.update_status(f"Error generating collision avoidance path: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
        
    def pick_and_place_with_collision_avoidance(self):
        if not hasattr(self, 'robot_square_positions') or not self.robot_square_positions:
            self.update_status("No squares mapped to robot coordinates")
            return
        
        # Access the robot_gui
        if not hasattr(self, 'robot_gui') or self.robot_gui is None:
            self.update_status("Robot control not available - direct reference not set")
            return
        
        robot_gui = self.robot_gui
        self.update_status("Found robot control interface. Proceeding with collision-aware pick and place.")
        
        # Import required modules
        import copy
        import rospy
        import moveit_commander
        from geometry_msgs.msg import Pose
        import pandas as pd
        import time

        # Initialize MoveIt (if not already initialized)
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            group = moveit_commander.MoveGroupCommander("manipulator")
            self.update_status("MoveIt initialized successfully")
        except Exception as e:
            self.update_status(f"Failed to initialize MoveIt: {str(e)}")
            return
        
        # Execute sequence for each square
        for i, (robot_x, robot_y, color) in enumerate(self.robot_square_positions):
            self.update_status(f"Processing {color} square at ({robot_x:.2f}, {robot_y:.2f}) with collision avoidance...")
            
            try:
                # 1. PICK PHASE
                # Generate pickup pose for the current square
                pickup_pose = self.calculate_pickup_pose(robot_x, robot_y)
                
                # Execute the pick phase (manual movement to the square)
                self.update_status(f"Executing pick phase for {color} square...")
                pickup_success = self.execute_pick_phase(robot_gui, pickup_pose)
                
                if not pickup_success:
                    self.update_status(f"Failed to pick up {color} square")
                    continue
                
                # 2. COLLISION-AWARE PATH PLANNING
                # Generate collision-free path for this square
                self.update_status(f"Generating collision-free path for {color} square...")
                success, waypoints_df = self.generate_collision_avoidance_path(square_index=i)
                print("DEBUG waypoint: ", waypoints_df)
                
                if not success:
                    self.update_status(f"Failed to generate collision-free path for {color} square")
                    continue
                
                # 3. COLLISION-FREE MOVEMENT EXECUTION
                self.update_status(f"Executing collision-free path for {color} square...")
                
                # Capture the current tool orientation
                start_pose = group.get_current_pose().pose
                tool_ori = start_pose.orientation
                
                # Apply velocity/acceleration scaling for smoother timing
                group.set_max_velocity_scaling_factor(0.5)
                group.set_max_acceleration_scaling_factor(0.1)
                
                # Build Pose waypoints from the DataFrame
                waypoints = []
                # Flip the x and y when using cartesian waypoint (as in the original script)
                for _, row in waypoints_df.iterrows():
                    p = Pose()
                    p.position.x = -float(row['x'])
                    p.position.y = -float(row['y'])
                    p.position.z = float(row['z'])
                    # Inject constant orientation to lock wrist
                    p.orientation = copy.deepcopy(tool_ori)
                    waypoints.append(p)
                
                self.update_status(f"Planning path with {len(waypoints)} waypoints...")
                
                # Compute Cartesian path
                eef_step = 0.01  # 1 cm interpolation
                traj_plan, fraction = group.compute_cartesian_path(
                    waypoints,
                    eef_step
                )
                
                if fraction < 0.99:
                    self.update_status(
                        f"Warning: Only {fraction*100:.1f}% of the Cartesian path was planned. "
                        "Check waypoint reachability."
                    )
                    if fraction < 0.7:  # If less than 70% of path is valid, abort
                        self.update_status(f"Path planning failed: insufficient coverage")
                        # self.execute_release_phase(robot_gui)
                        continue
                
                # Time-parameterize the trajectory
                current_state = group.get_current_state()
                timed_plan = group.retime_trajectory(
                    current_state,
                    traj_plan,
                    velocity_scaling_factor=0.1
                )
                
                # Execute the collision-free path
                self.update_status("Executing collision-free movement...")
                execution_success = group.execute(timed_plan, wait=True)
                
                if not execution_success:
                    self.update_status(f"Failed to execute collision-free path for {color} square")
                    continue
                
                    
            except Exception as e:
                self.update_status(f"Error during collision-aware pick and place: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try to safely release if we're holding something
                # self.execute_release_phase(robot_gui)
        
        # Return to a safe position when done
        self.update_status("Returning to camera capture position...")
        capture_image_pose = [-62.71, -88.97, -31.19, -149.76, 89.85, 27.15]
        robot_gui.set_joint_values(capture_image_pose)
        robot_gui.move_robot()
        
        self.update_status("Pick and place with collision avoidance completed")
    
class UR3eGUITab:
    def __init__(self, parent):
        self.parent = parent
                # Initialize ROS controller in a separate try-except block
        try:
            self.controller = None  # Initialize to None first
            self.init_status_label = ttk.Label(self.parent, text="Initializing robot controller...", foreground="blue")
            self.init_status_label.pack(pady=10)
            self.parent.update()  # Update the UI to show the initializing message
            
            # Initialize the controller
            self.controller = UR3eRealRobotController()
            
            # Initialize gripper publisher
            self.gripper_publisher = rospy.Publisher('/onrobot/joint_position_controller/command', 
                                                    Float64, queue_size=10)
            
            self.init_status_label.config(text="Robot controller initialized successfully!", foreground="green")
        except Exception as e:
            self.init_status_label.config(text=f"Failed to initialize controller: {str(e)}", foreground="red")
            messagebox.showerror("Initialization Error", f"Failed to initialize robot controller: {str(e)}")

        # Create main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Create joint control frame
        joint_frame = ttk.LabelFrame(main_frame, text="Joint Control")
        joint_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Create variables for joint values
        self.joint_values = []
        self.joint_sliders = []
        self.joint_entry_deg = []
        self.joint_entry_rad = []
        
        # Create radio button for unit selection
        self.unit_var = tk.StringVar(value="deg")
        unit_frame = ttk.Frame(joint_frame)
        unit_frame.pack(pady=5)
        ttk.Radiobutton(unit_frame, text="Degrees", variable=self.unit_var, value="deg").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(unit_frame, text="Radians", variable=self.unit_var, value="rad").pack(side=tk.LEFT, padx=10)
        
        # Create joint controls
        for i in range(6):
            joint_var = tk.DoubleVar(value=0)
            self.joint_values.append(joint_var)
            
            joint_row = ttk.Frame(joint_frame)
            joint_row.pack(pady=5, fill=tk.X)
            
            # Joint label
            ttk.Label(joint_row, text=f"q{i+1}:", width=3).pack(side=tk.LEFT, padx=5)
            
            # Joint slider
            slider = ttk.Scale(joint_row, from_=-180, to=180, orient=tk.HORIZONTAL, 
                              variable=joint_var, command=lambda val, idx=i: self.update_entries(idx))
            slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.joint_sliders.append(slider)
            
            # Entry in degrees
            deg_entry = ttk.Entry(joint_row, width=8)
            deg_entry.pack(side=tk.LEFT, padx=5)
            deg_entry.bind("<Return>", lambda event, idx=i, unit="deg": self.update_from_entry(idx, unit))
            deg_entry.bind("<FocusOut>", lambda event, idx=i, unit="deg": self.update_from_entry(idx, unit))
            self.joint_entry_deg.append(deg_entry)
            ttk.Label(joint_row, text="°").pack(side=tk.LEFT)
            
            # Entry in radians
            rad_entry = ttk.Entry(joint_row, width=8)
            rad_entry.pack(side=tk.LEFT, padx=5)
            rad_entry.bind("<Return>", lambda event, idx=i, unit="rad": self.update_from_entry(idx, unit))
            rad_entry.bind("<FocusOut>", lambda event, idx=i, unit="rad": self.update_from_entry(idx, unit))
            self.joint_entry_rad.append(rad_entry)
            ttk.Label(joint_row, text="rad").pack(side=tk.LEFT)
        
        # Update entries with initial values
        for i in range(6):
            self.update_entries(i)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10, fill=tk.X)
        
        # Current position button
        ttk.Button(control_frame, text="Get Current Position", 
                   command=self.get_current_position).pack(side=tk.LEFT, padx=5)
        
        # Home position button
        ttk.Button(control_frame, text="Home Position", 
                   command=self.go_to_home).pack(side=tk.LEFT, padx=5)
        
        # Move button
        self.move_button = ttk.Button(control_frame, text="Move Robot", 
                                     command=self.move_robot)
        self.move_button.pack(side=tk.RIGHT, padx=5)
        
        # Velocity scale
        velocity_frame = ttk.Frame(main_frame)
        velocity_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(velocity_frame, text="Velocity scaling:").pack(side=tk.LEFT, padx=5)
        self.velocity_var = tk.DoubleVar(value=0.3)  # Default 30%
        velocity_scale = ttk.Scale(velocity_frame, from_=0.05, to=1.0, 
                                  orient=tk.HORIZONTAL, variable=self.velocity_var)
        velocity_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        velocity_label = ttk.Label(velocity_frame, textvariable=tk.StringVar())
        velocity_label.pack(side=tk.LEFT, padx=5)
        
        # Update velocity label when slider moves
        def update_velocity_label(*args):
            velocity_label.config(text=f"{self.velocity_var.get():.2f} (max: 1.0)")
        
        self.velocity_var.trace_add("write", update_velocity_label)
        update_velocity_label()  # Initialize the label
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(pady=10, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, padding=10)
        status_label.pack(fill=tk.X)
        
        # Position display frame
        position_frame = ttk.LabelFrame(main_frame, text="Current Position")
        position_frame.pack(pady=10, fill=tk.X)
        
        self.position_var = tk.StringVar(value="No data")
        position_label = ttk.Label(position_frame, textvariable=self.position_var, padding=10)
        position_label.pack(fill=tk.X)
        
        # Preset positions frame
        preset_frame = ttk.LabelFrame(main_frame, text="Preset Positions")
        preset_frame.pack(pady=10, fill=tk.X)
        
        preset_buttons_frame = ttk.Frame(preset_frame)
        preset_buttons_frame.pack(pady=5, fill=tk.X)
        
        # Define preset positions
        self.presets = {
            "Home": [pi/2, -pi/2, 0, -pi/2, 0, 0],
            "Capture image position": [np.deg2rad(-62.71), np.deg2rad(-88.97), np.deg2rad(-31.19), 
                           np.deg2rad(-149.76), np.deg2rad(89.85), np.deg2rad(27.15)],
            "L Shape": [np.deg2rad(90), np.deg2rad(-90), np.deg2rad(90), 
                       np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(0)],
            "Drop-off Pose 1": [np.deg2rad(45), np.deg2rad(-45), np.deg2rad(90), 
                               np.deg2rad(-45), np.deg2rad(-90), np.deg2rad(0)],
            "Drop-off Pose 2": [np.deg2rad(0), np.deg2rad(-60), np.deg2rad(100), 
                               np.deg2rad(-40), np.deg2rad(0), np.deg2rad(90)],
            "Drop-off Pose 3": [np.deg2rad(-45), np.deg2rad(-30), np.deg2rad(80), 
                               np.deg2rad(-50), np.deg2rad(45), np.deg2rad(45)],
            # Added the three new pick-up positions
            "Pick up pos 1": [np.deg2rad(30), np.deg2rad(-60), np.deg2rad(90), 
                             np.deg2rad(-120), np.deg2rad(60), np.deg2rad(0)],
            "Pick up pos 2": [np.deg2rad(0), np.deg2rad(-75), np.deg2rad(85), 
                             np.deg2rad(-100), np.deg2rad(90), np.deg2rad(45)],
            "Pick up pos 3": [np.deg2rad(-30), np.deg2rad(-45), np.deg2rad(70), 
                             np.deg2rad(-115), np.deg2rad(-60), np.deg2rad(30)]
        }
        
        # File path for saving custom positions
        self.custom_positions_file = "ur3e_custom_positions.txt"
        
        # Try to load custom positions if the file exists
        try:
            self.load_custom_positions()
        except Exception as e:
            print(f"Could not load custom positions: {e}")
            
        # Create buttons for presets - use a two-row layout for better organization
        row1_frame = ttk.Frame(preset_buttons_frame)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(preset_buttons_frame)
        row2_frame.pack(fill=tk.X, pady=2)
            
        # First row presets
        first_row = ["Home", "Capture image position", "L Shape", "Drop-off Pose 1", "Drop-off Pose 2"]
        for name in first_row:
            ttk.Button(row1_frame, text=name,
                      command=lambda pos=self.presets[name], name=name: self.set_preset(pos, name)).pack(side=tk.LEFT, padx=5)
        
        # Second row presets
        second_row = ["Drop-off Pose 3", "Pick up pos 1", "Pick up pos 2", "Pick up pos 3"]
        for name in second_row:
            ttk.Button(row2_frame, text=name,
                      command=lambda pos=self.presets[name], name=name: self.set_preset(pos, name)).pack(side=tk.LEFT, padx=5)
        
        # Create save position frame
        save_frame = ttk.LabelFrame(main_frame, text="Save Custom Positions")
        save_frame.pack(pady=10, fill=tk.X)
        
        save_pos_frame = ttk.Frame(save_frame)
        save_pos_frame.pack(pady=5, fill=tk.X)
        
        # Position name entry
        ttk.Label(save_pos_frame, text="Position Name:").pack(side=tk.LEFT, padx=5)
        self.position_name_var = tk.StringVar()
        position_name_entry = ttk.Entry(save_pos_frame, width=15, textvariable=self.position_name_var)
        position_name_entry.pack(side=tk.LEFT, padx=5)
        
        # Position type selection
        ttk.Label(save_pos_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.position_type_var = tk.StringVar(value="Pick up")
        ttk.Radiobutton(save_pos_frame, text="Pick up", variable=self.position_type_var, 
                        value="Pick up").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(save_pos_frame, text="Drop-off", variable=self.position_type_var, 
                        value="Drop-off").pack(side=tk.LEFT, padx=5)
        
        # Save button
        ttk.Button(save_pos_frame, text="Save Current Position", 
                   command=self.save_current_position).pack(side=tk.RIGHT, padx=20)
        
        # Custom positions frame
        custom_frame = ttk.LabelFrame(main_frame, text="Custom Positions")
        custom_frame.pack(pady=10, fill=tk.X)
        
        self.custom_positions_frame = ttk.Frame(custom_frame)
        self.custom_positions_frame.pack(pady=5, fill=tk.X)
        
        # Initialize custom positions buttons (will be populated when positions are loaded/saved)
        self.update_custom_position_buttons()
        
        # Add gripper control section
        self.add_gripper_controls(main_frame)
        
        # Check if controller was initialized properly
        if self.controller is None:
            self.status_var.set("Robot controller not initialized. Check ROS connection.")
            self.disable_controls()
            
    def disable_controls(self):
        """Disable all control elements when controller not available"""
        for slider in self.joint_sliders:
            slider.state(["disabled"])
        for entry in self.joint_entry_deg + self.joint_entry_rad:
            entry.state(["disabled"])
        self.move_button.state(["disabled"])
        
        # Also disable gripper controls if they exist
        if hasattr(self, 'finger_width_var'):
            for child in self.parent.winfo_children():
                if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Gripper Control":
                    for subchild in child.winfo_children():
                        for widget in subchild.winfo_children():
                            if isinstance(widget, (ttk.Button, ttk.Entry)):
                                widget.state(["disabled"])
    
    def update_entries(self, index):
        """Update text entries when slider is moved"""
        value_deg = self.joint_values[index].get()
        value_rad = np.deg2rad(value_deg)
        
        self.joint_entry_deg[index].delete(0, tk.END)
        self.joint_entry_deg[index].insert(0, f"{value_deg:.2f}")
        
        self.joint_entry_rad[index].delete(0, tk.END)
        self.joint_entry_rad[index].insert(0, f"{value_rad:.4f}")
    
    def update_from_entry(self, index, unit):
        """Update slider and other entry when one entry is changed"""
        try:
            if unit == "deg":
                value = float(self.joint_entry_deg[index].get())
                self.joint_values[index].set(value)
                
                value_rad = np.deg2rad(value)
                self.joint_entry_rad[index].delete(0, tk.END)
                self.joint_entry_rad[index].insert(0, f"{value_rad:.4f}")
            else:  # unit == "rad"
                value_rad = float(self.joint_entry_rad[index].get())
                value_deg = np.rad2deg(value_rad)
                
                self.joint_values[index].set(value_deg)
                self.joint_entry_deg[index].delete(0, tk.END)
                self.joint_entry_deg[index].insert(0, f"{value_deg:.2f}")
        except ValueError:
            # Reset to current value if input is invalid
            self.update_entries(index)
    
    def get_joint_values_in_radians(self):
        """Get all joint values in radians"""
        if self.unit_var.get() == "deg":
            return [np.deg2rad(val.get()) for val in self.joint_values]
        else:
            return [float(entry.get()) for entry in self.joint_entry_rad]
    
    def move_robot(self):
        """Send movement command to the robot"""
        if self.controller is None:
            messagebox.showerror("Error", "Robot controller not initialized.")
            return
        
        try:
            # Get joint values in radians
            joint_values = self.get_joint_values_in_radians()
            
            # Get velocity scaling
            velocity = self.velocity_var.get()
            
            # Update status
            self.status_var.set(f"Moving robot to joint position... (velocity: {velocity:.2f})")
            self.parent.update()  # Force UI update
            
            # Execute the move
            success = self.controller.go_to_joint_positions(joint_values, velocity_scaling=velocity)
            
            if success:
                self.status_var.set("Robot movement completed successfully!")
                # Update position display after move completes
                self.get_current_position()
            else:
                self.status_var.set("Robot movement failed or position tolerance not achieved.")
                messagebox.showwarning("Movement Issue", "Robot movement failed or position tolerance not achieved.")
        
        except Exception as e:
            self.status_var.set(f"Error moving robot: {str(e)}")
            messagebox.showerror("Movement Error", f"Error moving robot: {str(e)}")
    
    def get_current_position(self):
        """Get and display current robot position"""
        if self.controller is None:
            messagebox.showerror("Error", "Robot controller not initialized.")
            return
        
        try:
            # Get current joint values
            current_joints = self.controller.get_current_joint_values()
            
            # Update display
            joints_deg = [np.rad2deg(val) for val in current_joints]
            position_text = f"q1={joints_deg[0]:.2f}°, q2={joints_deg[1]:.2f}°, q3={joints_deg[2]:.2f}°, " \
                          f"q4={joints_deg[3]:.2f}°, q5={joints_deg[4]:.2f}°, q6={joints_deg[5]:.2f}°"
            self.position_var.set(position_text)
            
            # Update sliders and entries
            for i, value in enumerate(joints_deg):
                self.joint_values[i].set(value)
                self.update_entries(i)
            
            self.status_var.set("Current position retrieved.")
        
        except Exception as e:
            self.status_var.set(f"Error getting current position: {str(e)}")
            messagebox.showerror("Position Error", f"Error getting current position: {str(e)}")
    
    def go_to_home(self):
        """Move robot to home position"""
        if self.controller is None:
            messagebox.showerror("Error", "Robot controller not initialized.")
            return
        
        try:
            # Confirm movement
            if not messagebox.askyesno("Confirm Movement", 
                                     "Move robot to home position?"):
                return
            
            # Home position
            home = [pi/2, -pi/2, 0, -pi/2, 0, 0]
            
            # Update status
            self.status_var.set("Moving robot to home position...")
            self.parent.update()  # Force UI update
            
            # Execute the move
            velocity = self.velocity_var.get()
            success = self.controller.go_to_joint_positions(home, velocity_scaling=velocity)
            
            if success:
                self.status_var.set("Robot moved to home position successfully!")
                # Update GUI with home position values
                home_deg = [np.rad2deg(val) for val in home]
                for i, value in enumerate(home_deg):
                    self.joint_values[i].set(value)
                    self.update_entries(i)
                # Update position display
                self.get_current_position()
            else:
                self.status_var.set("Failed to move to home position.")
                messagebox.showwarning("Movement Issue", "Failed to move to home position.")
        
        except Exception as e:
            self.status_var.set(f"Error moving to home: {str(e)}")
            messagebox.showerror("Movement Error", f"Error moving to home: {str(e)}")
    
    def set_preset(self, position, name):
        """Set joint values to a preset position"""
        if self.controller is None and name != "Custom":
            messagebox.showerror("Error", "Robot controller not initialized.")
            return
        
        try:
            # Update GUI with preset values
            position_deg = [np.rad2deg(val) for val in position]
            for i, value in enumerate(position_deg):
                self.joint_values[i].set(value)
                self.update_entries(i)
            
            self.status_var.set(f"Preset '{name}' loaded. Click 'Move Robot' to execute.")
        
        except Exception as e:
            self.status_var.set(f"Error setting preset: {str(e)}")
            messagebox.showerror("Preset Error", f"Error setting preset: {str(e)}")
     
    def save_current_position(self):
        """Save the current joint position with a custom name"""
        if self.controller is None:
            messagebox.showerror("Error", "Robot controller not initialized.")
            return
            
        # Get name for the position
        name = self.position_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name for the position.")
            return
            
        # Get position type
        position_type = self.position_type_var.get()
        full_name = f"{position_type} {name}"
            
        # Get current joint values
        try:
            # Use the values from the sliders
            joint_values = self.get_joint_values_in_radians()
                
            # Save to presets dictionary
            self.presets[full_name] = joint_values
                
            # Save to file
            self.save_custom_positions()
                
            # Update GUI
            self.update_custom_position_buttons()
            self.status_var.set(f"Position '{full_name}' saved successfully!")
            
        except Exception as e:
            self.status_var.set(f"Error saving position: {str(e)}")
            messagebox.showerror("Save Error", f"Error saving position: {str(e)}")
    
    def save_custom_positions(self):
        """Save custom positions to file"""
        try:
            with open(self.custom_positions_file, 'w') as f:
                for name, position in self.presets.items():
                    # Skip built-in presets
                    if name in ["Home", "Capture image position", "L Shape"]:
                        continue
                        
                    # Save as name,joint1,joint2,joint3,joint4,joint5,joint6
                    position_str = ",".join([name] + [str(val) for val in position])
                    f.write(position_str + "\n")
                    
            return True
        except Exception as e:
            print(f"Error saving custom positions: {e}")
            return False
    
    def update_custom_position_buttons(self):
        # Clear existing buttons
        for widget in self.custom_positions_frame.winfo_children():
            widget.destroy()
            
        # Create two rows for custom positions
        custom_pickup_frame = ttk.Frame(self.custom_positions_frame)
        custom_pickup_frame.pack(fill=tk.X, pady=2)
        custom_dropoff_frame = ttk.Frame(self.custom_positions_frame)
        custom_dropoff_frame.pack(fill=tk.X, pady=2)
        
        # Sort positions by name
        pickup_positions = []
        dropoff_positions = []
        
        for name in sorted(self.presets.keys()):
            # Skip built-in presets
            if name in ["Home", "Capture image position", "L Shape"]:
                continue
                
            if name.startswith("Pick up"):
                pickup_positions.append(name)
            elif name.startswith("Drop-off"):
                dropoff_positions.append(name)
        
        # Add buttons for custom pick-up positions
        if pickup_positions:
            ttk.Label(custom_pickup_frame, text="Pick-up:").pack(side=tk.LEFT, padx=5)
            for name in pickup_positions:
                btn = ttk.Button(custom_pickup_frame, text=name.replace("Pick up ", ""),
                           command=lambda pos=self.presets[name], name=name: self.set_preset(pos, name))
                btn.pack(side=tk.LEFT, padx=5)
                # Add a delete button
                delete_btn = ttk.Button(custom_pickup_frame, text="X", width=2,
                                     command=lambda name=name: self.delete_custom_position(name))
                delete_btn.pack(side=tk.LEFT, padx=0)
        
        # Add buttons for custom drop-off positions
        if dropoff_positions:
            ttk.Label(custom_dropoff_frame, text="Drop-off:").pack(side=tk.LEFT, padx=5)
            for name in dropoff_positions:
                btn = ttk.Button(custom_dropoff_frame, text=name.replace("Drop-off ", ""),
                           command=lambda pos=self.presets[name], name=name: self.set_preset(pos, name))
                btn.pack(side=tk.LEFT, padx=5)
                # Add a delete button
                delete_btn = ttk.Button(custom_dropoff_frame, text="X", width=2,
                                     command=lambda name=name: self.delete_custom_position(name))
                delete_btn.pack(side=tk.LEFT, padx=0)
    
    def delete_custom_position(self, name):
        if messagebox.askyesno("Confirm Delete", f"Delete position '{name}'?"):
            try:
                # Remove from presets dictionary
                if name in self.presets:
                    del self.presets[name]
                
                # Save changes to file
                self.save_custom_positions()
                
                # Update GUI
                self.update_custom_position_buttons()
                self.status_var.set(f"Position '{name}' deleted.")
            except Exception as e:
                self.status_var.set(f"Error deleting position: {str(e)}")
                messagebox.showerror("Delete Error", f"Error deleting position: {str(e)}")
            
    def add_gripper_controls(self, parent_frame):
        # Create gripper control frame
        gripper_frame = ttk.LabelFrame(parent_frame, text="Gripper Control")
        gripper_frame.pack(pady=10, fill=tk.X)
        
        # Create controls inside the frame
        gripper_controls = ttk.Frame(gripper_frame)
        gripper_controls.pack(pady=10, padx=10, fill=tk.X)
        
        # Add label and entry for finger width
        ttk.Label(gripper_controls, text="Finger Width (m):").pack(side=tk.LEFT, padx=5)
        
        # Variable for finger width with default value
        self.finger_width_var = tk.StringVar(value="0.05")
        
        # Entry field for finger width
        finger_width_entry = ttk.Entry(gripper_controls, textvariable=self.finger_width_var, width=10)
        finger_width_entry.pack(side=tk.LEFT, padx=5)
        
        # Add gripper control buttons
        gripper_buttons_frame = ttk.Frame(gripper_controls)
        gripper_buttons_frame.pack(side=tk.RIGHT, padx=5)
        
        # Predefined positions
        ttk.Button(gripper_buttons_frame, text="Open (0.1m)", 
                  command=lambda: self.set_gripper_width("0.1")).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(gripper_buttons_frame, text="Half (0.04m)", 
                  command=lambda: self.set_gripper_width("0.04")).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(gripper_buttons_frame, text="Close (0.056m)", 
                  command=lambda: self.set_gripper_width("0.056")).pack(side=tk.LEFT, padx=5)
        
    def set_gripper_width(self, width):
        self.finger_width_var.set(width)
    
    def send_gripper_command(self):
        if not hasattr(self, 'gripper_publisher'):
            messagebox.showerror("Error", "Gripper publisher not initialized.")
            return   
        try:
            # Get the width from the entry
            width_str = self.finger_width_var.get()
            width = float(width_str)
            
            # Validate width range (adjust these values based on your gripper's limits)
            if width < 0.0 or width > 0.15:
                messagebox.showwarning("Invalid Width", 
                                     "Gripper width should be between 0.0 and 0.15 meters.")
                return
                
            # Create and publish the message
            msg = Float64()
            msg.data = width
            self.gripper_publisher.publish(msg)
            
            # Update status
            self.status_var.set(f"Gripper command sent: width = {width:.3f}m")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for gripper width.")
        except Exception as e:
            self.status_var.set(f"Error sending gripper command: {str(e)}")
            messagebox.showerror("Gripper Command Error", f"Error: {str(e)}")
            
    def set_joint_values(self, joint_values):
        for i, value in enumerate(joint_values):
            if i < len(self.joint_values):
                self.joint_values[i].set(value)
                self.update_entries(i)

    def set_preset_by_name(self, preset_name):
        if preset_name in self.presets:
            position = self.presets[preset_name]
            position_deg = [np.rad2deg(val) for val in position]
            self.set_joint_values(position_deg)
        else:
            print(f"Preset {preset_name} not found")
            
            
def main():
    root = tk.Tk()
    app = CombinedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()