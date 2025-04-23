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
# Import the specific modules from each application
# For shape detection GUI
try:
    from shape_color_detection import detect_shapes_and_colors
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
        self.shape_tab = ttk.Frame(self.notebook)
        self.camera_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.shape_tab, text="Shape Detection")
        self.notebook.add(self.robot_tab, text="Robot Control")
        self.notebook.add(self.camera_tab, text="Camera")
        
        # Initialize the shape detection tab
        self.init_shape_detection()
        
        # Initialize the robot control tab
        self.init_robot_control()
        
        self.init_camera()
        
        # Connect the tabs (add this part)
        self.shape_gui.combined_gui = self
        self.shape_gui.robot_gui = self.robot_gui
        
    def init_shape_detection(self):
        """Initialize the shape detection interface in its tab"""
        self.shape_gui = ShapeDetectionGUITab(self.shape_tab)

    def init_robot_control(self):
        """Initialize the robot control interface in its tab"""
        self.robot_gui = UR3eGUITab(self.robot_tab)
        
    def init_camera(self):
        """Initialize the camera tab"""
        self.camera_gui = CameraTab(self.camera_tab)
    

class CameraTab:
    def __init__(self, parent):
        """Initialize the camera capture tab"""
        self.parent = parent  # Store the parent widget
        self.root = parent.master.master  # Access to the root window
        
        # Initialize attributes
        self.pipeline = None
        self.is_capturing = False
        self.last_captured_image_path = None
        
        # Main frame
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="RealSense Camera Capture", font=("Arial", 16)).pack(side=tk.LEFT, padx=10)
        
        self.capture_btn = ttk.Button(controls_frame, text="Start Capture", command=self.toggle_capture)
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        self.single_capture_btn = ttk.Button(controls_frame, text="Take Single Picture", command=self.take_single_picture)
        self.single_capture_btn.pack(side=tk.LEFT, padx=10)
        
        # Output directory selection
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        ttk.Label(dir_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Capture interval
        interval_frame = ttk.Frame(frame)
        interval_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        ttk.Label(interval_frame, text="Capture Interval (seconds):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(interval_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.interval_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Image preview
        self.preview_frame = ttk.LabelFrame(frame, text="Camera Preview")
        self.preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Last captured image
        self.last_image_frame = ttk.LabelFrame(frame, text="Last Captured Image")
        self.last_image_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
        
        self.last_image_label = ttk.Label(self.last_image_frame)
        self.last_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Status frame
        status_frame = ttk.Frame(frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Camera ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
    
    def update_status(self, message):
        """Update the status message and print to console"""
        self.status_var.set(message)
        print(message)
    
    def select_output_dir(self):
        """Open directory dialog to select output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
            self.update_status(f"Output directory set to: {directory}")
    
    def toggle_capture(self):
        """Start or stop the continuous image capture"""
        if self.is_capturing:
            self.stop_capture()
        else:
            self.start_capture()
    
    def start_capture(self):
        """Start the continuous image capture thread"""
        try:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            
            self.is_capturing = True
            self.capture_btn.config(text="Stop Capture")
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            self.update_status("Camera capture started")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}")
            self.update_status(f"Error: {str(e)}")
    
    def stop_capture(self):
        """Stop the continuous image capture"""
        self.is_capturing = False
        
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            
        self.capture_btn.config(text="Start Capture")
        self.update_status("Camera capture stopped")
    
    def capture_loop(self):
        """Continuous image capture loop that runs in a separate thread"""
        count = 0
        
        try:
            while self.is_capturing:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Update preview
                self.update_preview(color_image)
                
                # Save an image based on the interval
                if count % max(1, int(30 * self.interval_var.get())) == 0:
                    self.save_image(color_image)
                
                count += 1
                time.sleep(1/30)  # Assuming 30 fps
        except Exception as e:
            # Use parent.after to schedule UI updates from the thread
            self.parent.after(0, lambda: messagebox.showerror("Capture Error", f"Error in capture loop: {str(e)}"))
            self.parent.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            self.parent.after(0, self.stop_capture)
    
    def take_single_picture(self):
        """Take a single picture from the camera"""
        try:
            if not self.pipeline:
                # Initialize RealSense pipeline just for one frame
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(config)
                
                # Wait for frames
                for _ in range(5):  # Wait for a few frames to stabilize
                    frames = pipeline.wait_for_frames()
                
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    raise Exception("Failed to get color frame")
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Save image
                self.save_image(color_image)
                
                # Stop pipeline
                pipeline.stop()
            else:
                # Use existing pipeline
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    raise Exception("Failed to get color frame")
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Save image
                self.save_image(color_image)
                
            self.update_status("Single picture captured successfully")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not take picture: {str(e)}")
            self.update_status(f"Error: {str(e)}")
    
    def save_image(self, image):
        """Save an image to the output directory"""
        try:
            # Create timestamp for filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"realsense_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir_var.get(), filename)
            
            # Save the image
            cv2.imwrite(filepath, image)
            
            # Update last captured image
            self.last_captured_image_path = filepath
            self.update_last_image(image)
            
            self.update_status(f"Saved image: {filename}")
            
        except Exception as e:
            # Use parent.after for thread safety
            self.parent.after(0, lambda: messagebox.showerror("Save Error", f"Could not save image: {str(e)}"))
            self.parent.after(0, lambda: self.update_status(f"Error: {str(e)}"))
    
    def update_preview(self, image):
        """Update the preview image in the UI"""
        try:
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display if needed
            h, w = image_rgb.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image_rgb = cv2.resize(image_rgb, new_size)
            
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update label
            self.preview_label.configure(image=image_tk)
            self.preview_label.image = image_tk  # Keep a reference
            
        except Exception as e:
            self.update_status(f"Error updating preview: {str(e)}")
    
    def update_last_image(self, image):
        """Update the last captured image in the UI"""
        try:
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display if needed
            h, w = image_rgb.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image_rgb = cv2.resize(image_rgb, new_size)
            
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update label
            self.last_image_label.configure(image=image_tk)
            self.last_image_label.image = image_tk  # Keep a reference
            
        except Exception as e:
            self.update_status(f"Error updating last image: {str(e)}")
            
class ShapeDetectionGUITab:
    def __init__(self, parent):
        self.parent = parent
        
        # Variables
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
        
        # Create frames
        self.create_frames()
        
        # Create widgets
        self.create_widgets()
        
        # Initialize status
        self.update_status("Ready. Please load an image.")
    
    def create_frames(self):
        # Main frame layout
        self.left_frame = ttk.Frame(self.parent, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.parent, padding="10", width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image display frame
        self.image_frame = ttk.Frame(self.left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Status bar
        self.status_frame = ttk.Frame(self.parent, padding="5")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_widgets(self):
        # Right panel controls
        ttk.Label(self.right_frame, text="Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Load image button
        ttk.Button(self.right_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Target selection
        ttk.Label(self.right_frame, text="Target Object:").pack(anchor=tk.W, pady=(10,0))
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
                self.right_frame, 
                text=text, 
                value=value, 
                variable=self.target_var,
                command=self.update_target
            ).pack(anchor=tk.W, padx=10)
        
        # ROI controls
        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(self.right_frame, text="Region of Interest (ROI)").pack(anchor=tk.W)
        
        ttk.Button(self.right_frame, text="Clear ROI", command=self.clear_roi).pack(fill=tk.X, pady=5)
        ttk.Label(self.right_frame, text="Instructions:").pack(anchor=tk.W, pady=(10,0))
        instructions = (
            "1. Load an image\n"
            "2. Draw ROI by dragging mouse\n"
            "3. Press Detect button\n"
            "4. View results"
        )
        ttk.Label(self.right_frame, text=instructions, wraplength=180).pack(anchor=tk.W, padx=10)
        
        # Detect button
        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        self.detect_button = ttk.Button(
            self.right_frame, 
            text="Detect Shapes", 
            command=self.detect_shapes,
            state=tk.DISABLED
        )
        self.detect_button.pack(fill=tk.X, pady=10)
        
        # Reset view button
        ttk.Button(self.right_frame, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=5)
        
        # In your create_widgets function
        self.pick_place_button = ttk.Button(
            self.right_frame,
            text="Pick and Place Squares",
            command=self.pick_and_place_squares,
            state=tk.DISABLED  # Initially disabled until squares are detected
        )
        self.pick_place_button.pack(fill=tk.X, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.update_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        
    def load_image(self):
        """Open a file dialog to select an image file"""
        from tkinter import filedialog
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
                self.show_image(self.display_image)
                
                self.update_status(f"Loaded image: {os.path.basename(file_path)}")
                self.detect_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")
    
    def show_image(self, image):
        """Display an image on the canvas"""
        # Convert the image to PIL format
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def start_roi(self, event):
        """Start ROI selection on mouse click"""
        if self.original_image is None:
            return
        
        # Get the canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        self.roi_start_x = int(x)
        self.roi_start_y = int(y)
        self.is_drawing_roi = True
        
        # Clear previous ROI
        self.canvas.delete("roi")
        
    def update_roi(self, event):
        """Update ROI rectangle as mouse moves"""
        if not self.is_drawing_roi:
            return
        
        # Get the canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        self.roi_end_x = int(x)
        self.roi_end_y = int(y)
        
        # Clear and redraw ROI
        self.canvas.delete("roi")
        self.canvas.create_rectangle(
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
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        self.roi_end_x = int(x)
        self.roi_end_y = int(y)
        
        # Ensure start coordinates are smaller than end coordinates
        if self.roi_start_x > self.roi_end_x:
            self.roi_start_x, self.roi_end_x = self.roi_end_x, self.roi_start_x
        
        if self.roi_start_y > self.roi_end_y:
            self.roi_start_y, self.roi_end_y = self.roi_end_y, self.roi_start_y
        
        # Redraw final ROI
        self.canvas.delete("roi")
        self.canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, 
            self.roi_end_x, self.roi_end_y, 
            outline="yellow", width=2, tags="roi"
        )
        
        roi_width = self.roi_end_x - self.roi_start_x
        roi_height = self.roi_end_y - self.roi_start_y
        self.update_status(f"ROI selected: ({self.roi_start_x}, {self.roi_start_y}) to ({self.roi_end_x}, {self.roi_end_y}), size: {roi_width}x{roi_height}")
    
    def clear_roi(self):
        """Clear the selected ROI"""
        self.canvas.delete("roi")
        self.roi_start_x = None
        self.roi_start_y = None
        self.roi_end_x = None
        self.roi_end_y = None
        
        if self.display_image is not None:
            self.show_image(self.display_image)
        
        self.update_status("ROI cleared")
    
    def detect_shapes(self):
        """Run shape detection on the selected ROI or the entire image"""
        if self.original_image is None:
            self.update_status("Please load an image first")
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
            self.show_image(display_result)
            
            self.update_status(f"Detection completed for target: {target}")
            # Map squares to robot coordinates
            if self.detected_squares:
                self.map_squares_to_robot_frame()
                self.pick_place_button.config(state=tk.NORMAL)
                self.update_status(f"Found {len(self.detected_squares)} squares. Ready for pick and place.")
            else:
                self.pick_place_button.config(state=tk.DISABLED)
                self.update_status("No squares detected.")
            
        except Exception as e:
            self.update_status(f"Error during detection: {str(e)}")
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
            self.show_image(self.display_image)
            
            # Redraw ROI if it exists
            if (self.roi_start_x is not None and self.roi_end_x is not None and 
                self.roi_start_y is not None and self.roi_end_y is not None):
                
                self.canvas.create_rectangle(
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
            pixel_x, pixel_y, color = square
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
    
    def pick_and_place_squares(self):
        """Execute pick and place sequence for all detected squares"""
        if not hasattr(self, 'robot_square_positions') or not self.robot_square_positions:
            self.update_status("No squares mapped to robot coordinates")
            return
        
        # Access the robot_gui through the direct reference we added in the CombinedGUI.__init__ method
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
            "red": [-148.39, -82.65, -111.35, -76.5, 89.96, 33.93],
            "blue": [-133.7, -88.60, -105.95, -75.93, 90.12, 49.25],
            "yellow": [-113.7, -121.09, -63.99, -85.32, 90.36, 68.52]
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
            gripper_delay = 1.5   # Delay after gripper operation
            # Define an intermediate/safe pose (slightly above the workspace)
            # These values are in degrees - adjust as needed for your robot setup
            intermediate_pose = [59.25, -89.96, 67.24, -67.32, -89.04, -30]  # Example safe position above the workspace
            capture_image_pose = [-62.71, -88.97, -31.19, 
                           -149.76, 89.85, 27.15]
            
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
            
            # 7. Move to pre-dropoff position
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
            
            # 10. Move back to pre-dropoff position
            self.update_status("Moving to post-dropoff position...")
            robot_gui.set_joint_values(pre_dropoff_pose)
            robot_gui.move_robot()
            
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
            
            # # Confirm movement
            # if not messagebox.askyesno("Confirm Movement", 
            #                          f"Move robot to:\nq1={np.rad2deg(joint_values[0]):.2f}°, "
            #                          f"q2={np.rad2deg(joint_values[1]):.2f}°, "
            #                          f"q3={np.rad2deg(joint_values[2]):.2f}°, "
            #                          f"q4={np.rad2deg(joint_values[3]):.2f}°, "
            #                          f"q5={np.rad2deg(joint_values[4]):.2f}°, "
            #                          f"q6={np.rad2deg(joint_values[5]):.2f}°\n"
            #                          f"with velocity scaling: {velocity:.2f}?"):
            #     return
            
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
    
    def load_custom_positions(self):
        """Load custom positions from file"""
        try:
            if not os.path.exists(self.custom_positions_file):
                return
                
            with open(self.custom_positions_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 7:  # name + 6 joint values
                        name = parts[0]
                        try:
                            position = [float(val) for val in parts[1:]]
                            self.presets[name] = position
                        except ValueError:
                            print(f"Invalid joint values for position {name}")
                            
            # Update the GUI
            self.update_custom_position_buttons()
            return True
        except Exception as e:
            print(f"Error loading custom positions: {e}")
            return False
    
    def update_custom_position_buttons(self):
        """Update the custom positions buttons in the GUI"""
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
        """Delete a custom position"""
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
        """Add gripper control section to the GUI"""
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
        
        # Button to send command
        ttk.Button(gripper_controls, text="Set Gripper Position", 
                  command=self.send_gripper_command).pack(side=tk.RIGHT, padx=20)
    
    def set_gripper_width(self, width):
        """Set the gripper width value in the entry field"""
        self.finger_width_var.set(width)
    
    def send_gripper_command(self):
        """Send the gripper command to ROS"""
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
                
            # # Confirm action
            # if not messagebox.askyesno("Confirm Gripper Movement", 
            #                           f"Set gripper width to {width:.3f} meters?"):
            #     return
                
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
        """Set joint values from an external source"""
        for i, value in enumerate(joint_values):
            if i < len(self.joint_values):
                self.joint_values[i].set(value)
                self.update_entries(i)

    def set_preset_by_name(self, preset_name):
        """Set a preset position by name"""
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