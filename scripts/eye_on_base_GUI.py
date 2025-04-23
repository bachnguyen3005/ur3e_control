import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageTk
import threading
import time
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# Import shape detection functionality
# Assuming shape_color_detection.py is in the same directory
try:
    from shape_color_detection import detect_shapes_and_colors
except ImportError:
    # Create a dummy function if the import fails
    def detect_shapes_and_colors(image, target, roi_offset=None):
        # Just return the original image with a text notification
        result = image.copy()
        cv2.putText(result, "Shape detection module not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

class IntegratedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Camera and Vision Processing")
        self.root.geometry("1200x800")
        
        # Create the notebook (tabs container)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Camera Capture")
        self.notebook.add(self.tab2, text="Shape Detection")
        self.notebook.add(self.tab3, text="Depth Extraction")
        self.notebook.add(self.tab4, text="Coordinate Calculation")
        
        # Shared variables between tabs
        self.last_captured_image_path = None
        self.current_image_path = None
        self.current_image = None
        self.depth_value = None
        self.pixel_x = None
        self.pixel_y = None
        
        # Initialize tabs
        self.init_camera_tab()
        self.init_shape_detection_tab()
        self.init_depth_tab()
        self.init_coordinate_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.update_status("Application started. Please select a tab to begin.")
        
        # RealSense pipeline variables
        self.pipeline = None
        self.is_capturing = False
        self.capture_thread = None
        
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_var.set(message)
        print(message)

    def update_pixel_coordinates(self, x, y):
        """Update pixel coordinates in the depth and coordinate tabs"""
        if hasattr(self, 'depth_x_var') and hasattr(self, 'depth_y_var'):
            self.depth_x_var.set(x)
            self.depth_y_var.set(y)
        
        if hasattr(self, 'coord_x_var') and hasattr(self, 'coord_y_var'):
            self.coord_x_var.set(x)
            self.coord_y_var.set(y)
            
    def update_depth_image(self, image):
        """Update the image in the depth tab"""
        if image is None:
            return
            
        self.depth_image = image.copy()
        
        # Convert to RGB for display
        display_image = cv2.cvtColor(self.depth_image, cv2.COLOR_BGR2RGB)
        
        # Resize for display if needed
        h, w = display_image.shape[:2]
        max_size = 600
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            display_image = cv2.resize(display_image, new_size)
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(display_image)
        self.depth_tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.depth_canvas.delete("all")
        self.depth_canvas.config(width=pil_image.width, height=pil_image.height)
        self.depth_canvas.create_image(0, 0, anchor=tk.NW, image=self.depth_tk_image)
        
        # Draw a marker at the current pixel location if set
        if self.pixel_x is not None and self.pixel_y is not None:
            # Scale coordinates if image was resized
            scale_x = pil_image.width / w
            scale_y = pil_image.height / h
            x = int(self.pixel_x * scale_x)
            y = int(self.pixel_y * scale_y)
            
            self.depth_canvas.create_oval(x-5, y-5, x+5, y+5, outline="red", width=2)
            self.depth_canvas.create_line(x-10, y, x+10, y, fill="red", width=2)
            self.depth_canvas.create_line(x, y-10, x, y+10, fill="red", width=2)
        
    # ===== TAB 1: Camera Capture =====
    def init_camera_tab(self):
        """Initialize the camera capture tab"""
        frame = ttk.Frame(self.tab1, padding="10")
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
            self.root.after(0, lambda: messagebox.showerror("Capture Error", f"Error in capture loop: {str(e)}"))
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            self.root.after(0, self.stop_capture)
            
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
            self.root.after(0, lambda: messagebox.showerror("Save Error", f"Could not save image: {str(e)}"))
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            
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
    
    # ===== TAB 2: Shape Detection =====
    def init_shape_detection_tab(self):
        """Initialize the shape detection tab"""
        # Adapt from shape_detection_gui.py
        
        # Main frame layout
        self.shape_left_frame = ttk.Frame(self.tab2, padding="10")
        self.shape_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.shape_right_frame = ttk.Frame(self.tab2, padding="10", width=200)
        self.shape_right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image display frame
        self.shape_image_frame = ttk.Frame(self.shape_left_frame)
        self.shape_image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.shape_canvas = tk.Canvas(self.shape_image_frame, bg="gray", highlightthickness=0)
        self.shape_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.shape_h_scrollbar = ttk.Scrollbar(self.shape_left_frame, orient=tk.HORIZONTAL, command=self.shape_canvas.xview)
        self.shape_h_scrollbar.pack(fill=tk.X)
        
        self.shape_v_scrollbar = ttk.Scrollbar(self.shape_image_frame, orient=tk.VERTICAL, command=self.shape_canvas.yview)
        self.shape_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.shape_canvas.configure(xscrollcommand=self.shape_h_scrollbar.set, yscrollcommand=self.shape_v_scrollbar.set)
        
        # Right panel controls
        ttk.Label(self.shape_right_frame, text="Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Load image button
        ttk.Button(self.shape_right_frame, text="Load Image", command=self.shape_load_image).pack(fill=tk.X, pady=5)
        
        # Use last captured image button
        ttk.Button(self.shape_right_frame, text="Use Last Captured", command=self.shape_use_last_captured).pack(fill=tk.X, pady=5)
        
        # Target selection
        ttk.Label(self.shape_right_frame, text="Target Object:").pack(anchor=tk.W, pady=(10,0))
        self.shape_target_var = tk.StringVar(value="all")
        targets = [
            ("All Objects", "all"),
            ("Red Circle", "red_circle"),
            ("Blue Triangle", "blue_triangle"),
            ("Blue Square", "blue_square"),
            ("Red Square", "red_square")
        ]
        
        for text, value in targets:
            ttk.Radiobutton(
                self.shape_right_frame, 
                text=text, 
                value=value, 
                variable=self.shape_target_var
            ).pack(anchor=tk.W, padx=10)
        
        # ROI controls
        ttk.Separator(self.shape_right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(self.shape_right_frame, text="Region of Interest (ROI)").pack(anchor=tk.W)
        
        ttk.Button(self.shape_right_frame, text="Clear ROI", command=self.shape_clear_roi).pack(fill=tk.X, pady=5)
        
        # Detect button
        ttk.Separator(self.shape_right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        self.shape_detect_button = ttk.Button(
            self.shape_right_frame, 
            text="Detect Shapes", 
            command=self.shape_detect_shapes,
            state=tk.DISABLED
        )
        self.shape_detect_button.pack(fill=tk.X, pady=10)
        
        # Reset view button
        ttk.Button(self.shape_right_frame, text="Reset View", command=self.shape_reset_view).pack(fill=tk.X, pady=5)
        
        # Canvas event bindings
        self.shape_canvas.bind("<ButtonPress-1>", self.shape_start_roi)
        self.shape_canvas.bind("<B1-Motion>", self.shape_update_roi)
        self.shape_canvas.bind("<ButtonRelease-1>", self.shape_end_roi)
        
        # ROI variables
        self.shape_roi_start_x = None
        self.shape_roi_start_y = None
        self.shape_roi_end_x = None
        self.shape_roi_end_y = None
        self.shape_is_drawing_roi = False
        
        # Image variables
        self.shape_original_image = None
        self.shape_display_image = None
        self.shape_result_image = None
        
        # Status label in shape tab
        self.shape_status_var = tk.StringVar()
        ttk.Label(self.shape_right_frame, textvariable=self.shape_status_var).pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
    def shape_load_image(self):
        """Open a file dialog to select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            try:
                # Load the image
                self.shape_original_image = cv2.imread(file_path)
                if self.shape_original_image is None:
                    self.update_status(f"Error: Could not load image from {file_path}")
                    return
                
                # Store the current image for other tabs
                self.current_image = self.shape_original_image.copy()
                
                # Convert to RGB for display
                self.shape_display_image = cv2.cvtColor(self.shape_original_image, cv2.COLOR_BGR2RGB)
                
                # Clear any existing ROI
                self.shape_clear_roi()
                
                # Display the image
                self.shape_show_image(self.shape_display_image)
                
                self.update_status(f"Loaded image: {os.path.basename(file_path)}")
                self.shape_detect_button.config(state=tk.NORMAL)
                
                # Also update the image in the depth tab
                self.update_depth_image(self.shape_original_image)
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")
                
    def shape_use_last_captured(self):
        """Use the last captured image from tab 1"""
        if self.last_captured_image_path and os.path.exists(self.last_captured_image_path):
            self.current_image_path = self.last_captured_image_path
            try:
                # Load the image
                self.shape_original_image = cv2.imread(self.last_captured_image_path)
                if self.shape_original_image is None:
                    self.update_status(f"Error: Could not load image from {self.last_captured_image_path}")
                    return
                
                # Store the current image for other tabs
                self.current_image = self.shape_original_image.copy()
                
                # Convert to RGB for display
                self.shape_display_image = cv2.cvtColor(self.shape_original_image, cv2.COLOR_BGR2RGB)
                
                # Clear any existing ROI
                self.shape_clear_roi()
                
                # Display the image
                self.shape_show_image(self.shape_display_image)
                
                self.update_status(f"Loaded last captured image: {os.path.basename(self.last_captured_image_path)}")
                self.shape_detect_button.config(state=tk.NORMAL)
                
                # Also update the image in the depth tab
                self.update_depth_image(self.shape_original_image)
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")
        else:
            self.update_status("No captured image available")
            messagebox.showinfo("No Image", "No captured image available. Please capture an image in the Camera tab first.")
            
    def shape_show_image(self, image):
        """Display an image on the canvas"""
        # Convert the image to PIL format
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        
        # Convert to PhotoImage
        self.shape_tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.shape_canvas.delete("all")
        self.shape_canvas.config(scrollregion=(0, 0, w, h))
        self.shape_canvas.create_image(0, 0, anchor=tk.NW, image=self.shape_tk_image)
        
    def shape_start_roi(self, event):
        """Start ROI selection on mouse click"""
        if self.shape_original_image is None:
            return
        
        # Get the canvas coordinates
        x = self.shape_canvas.canvasx(event.x)
        y = self.shape_canvas.canvasy(event.y)
        
        self.shape_roi_start_x = int(x)
        self.shape_roi_start_y = int(y)
        self.shape_is_drawing_roi = True
        
        # Clear previous ROI
        self.shape_canvas.delete("roi")
        
    def shape_update_roi(self, event):
        """Update ROI rectangle as mouse moves"""
        if not self.shape_is_drawing_roi:
            return
        
        # Get the canvas coordinates
        x = self.shape_canvas.canvasx(event.x)
        y = self.shape_canvas.canvasy(event.y)
        
        self.shape_roi_end_x = int(x)
        self.shape_roi_end_y = int(y)
        
        # Clear and redraw ROI
        self.shape_canvas.delete("roi")
        self.shape_canvas.create_rectangle(
            self.shape_roi_start_x, self.shape_roi_start_y, 
            self.shape_roi_end_x, self.shape_roi_end_y, 
            outline="yellow", width=2, tags="roi"
        )
        
    def shape_end_roi(self, event):
        """Finalize ROI selection on mouse release"""
        if not self.shape_is_drawing_roi:
            return
        
        self.shape_is_drawing_roi = False
        
        # Get the canvas coordinates
        x = self.shape_canvas.canvasx(event.x)
        y = self.shape_canvas.canvasy(event.y)
        
        self.shape_roi_end_x = int(x)
        self.shape_roi_end_y = int(y)
        
        # Ensure start coordinates are smaller than end coordinates
        if self.shape_roi_start_x > self.shape_roi_end_x:
            self.shape_roi_start_x, self.shape_roi_end_x = self.shape_roi_end_x, self.shape_roi_start_x
        
        if self.shape_roi_start_y > self.shape_roi_end_y:
            self.shape_roi_start_y, self.shape_roi_end_y = self.shape_roi_end_y, self.shape_roi_start_y
        
        # Redraw final ROI
        self.shape_canvas.delete("roi")
        self.shape_canvas.create_rectangle(
            self.shape_roi_start_x, self.shape_roi_start_y, 
            self.shape_roi_end_x, self.shape_roi_end_y, 
            outline="yellow", width=2, tags="roi"
        )
        
        roi_width = self.shape_roi_end_x - self.shape_roi_start_x
        roi_height = self.shape_roi_end_y - self.shape_roi_start_y
        self.update_status(f"ROI selected: ({self.shape_roi_start_x}, {self.shape_roi_start_y}) to ({self.shape_roi_end_x}, {self.shape_roi_end_y}), size: {roi_width}x{roi_height}")
        
        # Store selected coordinates for depth extraction
        self.pixel_x = int((self.shape_roi_start_x + self.shape_roi_end_x) / 2)
        self.pixel_y = int((self.shape_roi_start_y + self.shape_roi_end_y) / 2)
        
        # Update coordinate fields in depth tab and coordinate tab
        self.update_pixel_coordinates(self.pixel_x, self.pixel_y)
        
    def shape_clear_roi(self):
        """Clear the selected ROI"""
        self.shape_canvas.delete("roi")
        self.shape_roi_start_x = None
        self.shape_roi_start_y = None
        self.shape_roi_end_x = None
        self.shape_roi_end_y = None
        
        if self.shape_display_image is not None:
            self.shape_show_image(self.shape_display_image)
        
        self.update_status("ROI cleared")
    
    def shape_detect_shapes(self):
        """Run shape detection on the selected ROI or the entire image"""
        if self.shape_original_image is None:
            self.update_status("Please load an image first")
            return
        
        try:
            # Get the target object
            target = self.shape_target_var.get()
            
            # Create a copy of the original image
            image_to_process = self.shape_original_image.copy()
            
            # If ROI is selected, crop the image
            if (self.shape_roi_start_x is not None and self.shape_roi_end_x is not None and 
                self.shape_roi_start_y is not None and self.shape_roi_end_y is not None):
                
                # Ensure ROI is within image boundaries
                h, w = image_to_process.shape[:2]
                start_x = max(0, min(self.shape_roi_start_x, w-1))
                start_y = max(0, min(self.shape_roi_start_y, h-1))
                end_x = max(0, min(self.shape_roi_end_x, w-1))
                end_y = max(0, min(self.shape_roi_end_y, h-1))
                
                # Crop the image
                cropped = image_to_process[start_y:end_y, start_x:end_x]
                
                # Make sure we have a valid crop
                if cropped.size == 0:
                    self.update_status("Invalid ROI: zero size after crop")
                    return
                
                # Process the cropped image with ROI offset for coordinate calculation
                # Pass ROI offset to ensure coordinates are global
                cropped_result = detect_shapes_and_colors(cropped, target, roi_offset=(start_x, start_y))
                
                # Create a full-sized result image
                self.shape_result_image = image_to_process.copy()
                
                # Place the processed cropped region back into the full image
                self.shape_result_image[start_y:end_y, start_x:end_x] = cropped_result
                
                # Draw a rectangle around the ROI in the result image
                cv2.rectangle(self.shape_result_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
            else:
                # Process the entire image - no ROI offset needed
                self.shape_result_image = detect_shapes_and_colors(image_to_process, target)
            
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.shape_result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.shape_show_image(display_result)
            
            self.update_status(f"Detection completed for target: {target}")
            self.shape_status_var.set(f"Detection completed for target: {target}")
            
        except Exception as e:
            self.update_status(f"Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def shape_reset_view(self):
        """Reset to the original image view"""
        if self.shape_original_image is not None:
            # Convert to RGB for display
            self.shape_display_image = cv2.cvtColor(self.shape_original_image, cv2.COLOR_BGR2RGB)
            
            # Show original image
            self.shape_show_image(self.shape_display_image)
            
            # Redraw ROI if it exists
            if (self.shape_roi_start_x is not None and self.shape_roi_end_x is not None and 
                self.shape_roi_start_y is not None and self.shape_roi_end_y is not None):
                
                self.shape_canvas.create_rectangle(
                    self.shape_roi_start_x, self.shape_roi_start_y, 
                    self.shape_roi_end_x, self.shape_roi_end_y, 
                    outline="yellow", width=2, tags="roi"
                )
            
            self.update_status("View reset to original image")

    # ===== TAB 3: Depth Extraction =====
    def init_depth_tab(self):
        """Initialize the depth extraction tab"""
        # Create frames
        main_frame = ttk.Frame(self.tab3, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image display
        image_frame = ttk.LabelFrame(left_frame, text="Depth Image")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.depth_canvas = tk.Canvas(image_frame, bg="gray", highlightthickness=0)
        self.depth_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(right_frame, text="Depth Controls", padding=10)
        controls_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Pixel coordinates input
        ttk.Label(controls_frame, text="Pixel Coordinates:").pack(anchor=tk.W, pady=(10, 0))
        
        coord_frame = ttk.Frame(controls_frame)
        coord_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coord_frame, text="X:").pack(side=tk.LEFT, padx=5)
        self.depth_x_var = tk.IntVar()
        ttk.Entry(coord_frame, width=6, textvariable=self.depth_x_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(coord_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        self.depth_y_var = tk.IntVar()
        ttk.Entry(coord_frame, width=6, textvariable=self.depth_y_var).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Load Image", command=self.depth_load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Use Current", command=self.depth_use_current).pack(side=tk.LEFT, padx=5)
        
        # Depth extraction
        extract_frame = ttk.LabelFrame(right_frame, text="Depth Extraction", padding=10)
        extract_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        ttk.Button(extract_frame, text="Extract Depth", command=self.extract_depth).pack(fill=tk.X, pady=5)
        
        # Depth value display
        ttk.Label(extract_frame, text="Depth Value:").pack(anchor=tk.W, pady=(10, 0))
        
        self.depth_value_var = tk.StringVar(value="Not available")
        ttk.Label(extract_frame, textvariable=self.depth_value_var, font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        
        # Depth conversion
        ttk.Label(extract_frame, text="Depth (meters):").pack(anchor=tk.W, pady=(10, 0))
        
        self.depth_meters_var = tk.StringVar(value="Not available")
        ttk.Label(extract_frame, textvariable=self.depth_meters_var, font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        
        # Use in calculation button
        ttk.Button(extract_frame, text="Use in Coordinate Calculation", command=self.use_depth_in_calculation).pack(fill=tk.X, pady=10)
        
        # RealSense integration
        realsense_frame = ttk.LabelFrame(right_frame, text="RealSense Integration", padding=10)
        realsense_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        ttk.Button(realsense_frame, text="Connect to RealSense", command=self.connect_realsense).pack(fill=tk.X, pady=5)
        ttk.Button(realsense_frame, text="Get Depth from Camera", command=self.get_depth_from_camera).pack(fill=tk.X, pady=5)
        
        # Canvas click binding
        self.depth_canvas.bind("<Button-1>", self.depth_click)
        
        # Image and depth variables
        self.depth_image = None
        self.depth_tk_image = None
        
    def depth_click(self, event):
        """Handle click on the depth image"""
        if self.depth_image is None:
            return
            
        # Get canvas coordinates
        x, y = event.x, event.y
        
        # Convert to original image coordinates if resized
        h, w = self.depth_image.shape[:2]
        canvas_w = self.depth_canvas.winfo_width()
        canvas_h = self.depth_canvas.winfo_height()
        
        orig_x = int(x * (w / canvas_w))
        orig_y = int(y * (h / canvas_h))
        
        # Update pixel coordinates
        self.pixel_x = orig_x
        self.pixel_y = orig_y
        self.update_pixel_coordinates(orig_x, orig_y)
        
        # Redraw
        self.update_depth_image(self.depth_image)
        
        self.update_status(f"Selected pixel: ({orig_x}, {orig_y})")
        
    def depth_load_image(self):
        """Load an image for depth extraction"""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load the image
                image = cv2.imread(file_path)
                if image is None:
                    self.update_status(f"Error: Could not load image from {file_path}")
                    return
                
                # Store the image
                self.current_image = image.copy()
                self.current_image_path = file_path
                
                # Update the display
                self.update_depth_image(image)
                
                self.update_status(f"Loaded image: {os.path.basename(file_path)}")
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")
                
    def depth_use_current(self):
        """Use the current image from other tabs"""
        if self.current_image is not None:
            self.update_depth_image(self.current_image)
            self.update_status("Using current image for depth extraction")
        else:
            self.update_status("No current image available")
            messagebox.showinfo("No Image", "No image is currently loaded. Please load or capture an image first.")
            
    def extract_depth(self):
        """Extract depth value from the selected pixel"""
        if self.depth_image is None:
            self.update_status("Please load an image first")
            return
            
        if self.pixel_x is None or self.pixel_y is None:
            self.update_status("Please select a pixel first")
            return
            
        try:
            # In a real application, this would extract the actual depth
            # Since we don't have a real depth image, we'll simulate a depth value
            
            # For demo purposes, use a grayscale version to simulate depth
            gray = cv2.cvtColor(self.depth_image, cv2.COLOR_BGR2GRAY)
            depth_value = int(gray[self.pixel_y, self.pixel_x])
            
            # Convert to meters (simulated)
            depth_meters = depth_value / 1000.0
            
            # Store the values
            self.depth_value = depth_value
            
            # Update display
            self.depth_value_var.set(f"{depth_value} units")
            self.depth_meters_var.set(f"{depth_meters:.3f} m")
            
            self.update_status(f"Extracted depth at ({self.pixel_x}, {self.pixel_y}): {depth_value} units, {depth_meters:.3f} m")
            
        except Exception as e:
            self.update_status(f"Error extracting depth: {str(e)}")
            
    def use_depth_in_calculation(self):
        """Pass the extracted depth to the coordinate calculation tab"""
        if self.depth_value is None:
            self.update_status("Please extract depth first")
            messagebox.showinfo("No Depth", "Please extract a depth value first.")
            return
            
        # Set the depth in the coordinate tab
        if hasattr(self, 'coord_depth_var'):
            self.coord_depth_var.set(self.depth_value / 1000.0)  # Convert to meters
            self.update_status("Depth value passed to coordinate calculation")
            
            # Switch to the coordinate tab
            self.notebook.select(self.tab4)
        
    def connect_realsense(self):
        """Connect to a RealSense camera for depth data"""
        try:
            # Initialize RealSense pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            pipeline.start(config)
            
            # Get a single frame to confirm connection
            frames = pipeline.wait_for_frames()
            
            # Stop streaming
            pipeline.stop()
            
            self.update_status("Successfully connected to RealSense camera")
            messagebox.showinfo("Connection Success", "Successfully connected to RealSense camera")
            
        except Exception as e:
            self.update_status(f"Error connecting to RealSense: {str(e)}")
            messagebox.showerror("Connection Error", f"Could not connect to RealSense camera: {str(e)}")
            
    def get_depth_from_camera(self):
        """Get a depth frame from the RealSense camera"""
        try:
            # Initialize RealSense pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = pipeline.start(config)
            
            # Wait for frames
            for _ in range(5):  # Wait for a few frames to stabilize
                frames = pipeline.wait_for_frames()
            
            # Get color and depth frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                raise Exception("Failed to get frames from camera")
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            # Store color image for display
            self.current_image = color_image.copy()
            self.update_depth_image(color_image)
            
            # Extract depth at selected pixel if available
            if self.pixel_x is not None and self.pixel_y is not None:
                try:
                    depth_value = depth_image[self.pixel_y, self.pixel_x]
                    depth_meters = depth_value * depth_scale
                    
                    # Store values
                    self.depth_value = depth_value
                    
                    # Update display
                    self.depth_value_var.set(f"{depth_value} units")
                    self.depth_meters_var.set(f"{depth_meters:.3f} m")
                    
                    self.update_status(f"Extracted depth at ({self.pixel_x}, {self.pixel_y}): {depth_value} units, {depth_meters:.3f} m")
                    
                except Exception as e:
                    self.update_status(f"Error getting depth at pixel: {str(e)}")
            
            # Stop streaming
            pipeline.stop()
            
        except Exception as e:
            self.update_status(f"Error getting depth from camera: {str(e)}")
            messagebox.showerror("Camera Error", f"Could not get depth from camera: {str(e)}")

    # ===== TAB 4: Coordinate Calculation =====
    def init_coordinate_tab(self):
        """Initialize the coordinate calculation tab"""
        # Create main frame
        main_frame = ttk.Frame(self.tab4, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera parameters frame
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Parameters", padding=10)
        camera_frame.pack(fill=tk.X, pady=10)
        
        # Input frame for extrinsics
        extrinsics_frame = ttk.Frame(camera_frame)
        extrinsics_frame.pack(fill=tk.X, pady=5)
        
        # Translation parameters
        ttk.Label(extrinsics_frame, text="Translation (x, y, z):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        trans_frame = ttk.Frame(extrinsics_frame)
        trans_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(trans_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.trans_x_var = tk.DoubleVar(value=-0.02949)
        ttk.Entry(trans_frame, width=10, textvariable=self.trans_x_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(trans_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.trans_y_var = tk.DoubleVar(value=-0.871109)
        ttk.Entry(trans_frame, width=10, textvariable=self.trans_y_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(trans_frame, text="Z:").pack(side=tk.LEFT, padx=2)
        self.trans_z_var = tk.DoubleVar(value=0.51042)
        ttk.Entry(trans_frame, width=10, textvariable=self.trans_z_var).pack(side=tk.LEFT, padx=2)
        
        # Rotation parameters (quaternion)
        ttk.Label(extrinsics_frame, text="Rotation (quaternion):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        rot_frame = ttk.Frame(extrinsics_frame)
        rot_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(rot_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.rot_x_var = tk.DoubleVar(value=-0.184867)
        ttk.Entry(rot_frame, width=10, textvariable=self.rot_x_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(rot_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.rot_y_var = tk.DoubleVar(value=0.17938345)
        ttk.Entry(rot_frame, width=10, textvariable=self.rot_y_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(rot_frame, text="Z:").pack(side=tk.LEFT, padx=2)
        self.rot_z_var = tk.DoubleVar(value=0.6766299)
        ttk.Entry(rot_frame, width=10, textvariable=self.rot_z_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(rot_frame, text="W:").pack(side=tk.LEFT, padx=2)
        self.rot_w_var = tk.DoubleVar(value=0.689795)
        ttk.Entry(rot_frame, width=10, textvariable=self.rot_w_var).pack(side=tk.LEFT, padx=2)
        
        # Buttons for preset values
        preset_frame = ttk.Frame(camera_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="Load Default Extrinsics", command=self.load_default_extrinsics).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Save Current Extrinsics", command=self.save_extrinsics).pack(side=tk.LEFT, padx=5)
        
        # Pixel coordinates and depth frame
        pixel_frame = ttk.LabelFrame(main_frame, text="Pixel Coordinates and Depth", padding=10)
        pixel_frame.pack(fill=tk.X, pady=10)
        
        # Pixel coordinates
        coord_frame = ttk.Frame(pixel_frame)
        coord_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coord_frame, text="Pixel X:").pack(side=tk.LEFT, padx=5)
        self.coord_x_var = tk.IntVar(value=260)
        ttk.Entry(coord_frame, width=6, textvariable=self.coord_x_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(coord_frame, text="Pixel Y:").pack(side=tk.LEFT, padx=5)
        self.coord_y_var = tk.IntVar(value=435)
        ttk.Entry(coord_frame, width=6, textvariable=self.coord_y_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(coord_frame, text="Depth (meters):").pack(side=tk.LEFT, padx=5)
        self.coord_depth_var = tk.DoubleVar(value=0.576)
        ttk.Entry(coord_frame, width=8, textvariable=self.coord_depth_var).pack(side=tk.LEFT, padx=5)
        
        # Button to use values from depth tab
        ttk.Button(pixel_frame, text="Use Values from Depth Tab", command=self.use_depth_values).pack(anchor=tk.W, pady=5)
        
        # Calculation frame
        calc_frame = ttk.LabelFrame(main_frame, text="3D Coordinate Calculation", padding=10)
        calc_frame.pack(fill=tk.X, pady=10)
        
        # Calculate button
        ttk.Button(calc_frame, text="Calculate Robot Coordinates", command=self.calculate_coordinates).pack(fill=tk.X, pady=5)
        
        # Results frame
        result_frame = ttk.Frame(calc_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        # X coordinate
        ttk.Label(result_frame, text="Robot X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.robot_x_var = tk.StringVar(value="Not calculated")
        ttk.Label(result_frame, textvariable=self.robot_x_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Y coordinate
        ttk.Label(result_frame, text="Robot Y:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.robot_y_var = tk.StringVar(value="Not calculated")
        ttk.Label(result_frame, textvariable=self.robot_y_var, font=("Arial", 12, "bold")).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Z coordinate
        ttk.Label(result_frame, text="Robot Z:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.robot_z_var = tk.StringVar(value="Not calculated")
        ttk.Label(result_frame, textvariable=self.robot_z_var, font=("Arial", 12, "bold")).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Copy to clipboard button
        ttk.Button(calc_frame, text="Copy Coordinates to Clipboard", command=self.copy_coordinates).pack(fill=tk.X, pady=5)
        
        # Camera intrinsics (fixed for simplicity)
        self.camera_intrinsics = {
            'height': 480,
            'width': 640,
            'distortion_model': 'plumb_bob',
            'D': [0.0, 0.0, 0.0, 0.0, 0.0],
            'K': [606.1439208984375, 0.0, 319.3987731933594, 
                0.0, 604.884033203125, 254.05661010742188, 
                0.0, 0.0, 1.0],
            'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'P': [606.1439208984375, 0.0, 319.3987731933594, 0.0,
                0.0, 604.884033203125, 254.05661010742188, 0.0,
                0.0, 0.0, 1.0, 0.0]
        }
        
    def load_default_extrinsics(self):
        """Load default extrinsic parameters"""
        self.trans_x_var.set(-0.02949)
        self.trans_y_var.set(-0.871109)
        self.trans_z_var.set(0.51042)
        
        self.rot_x_var.set(-0.184867)
        self.rot_y_var.set(0.17938345)
        self.rot_z_var.set(0.6766299)
        self.rot_w_var.set(0.689795)
        
        self.update_status("Default extrinsic parameters loaded")
        
    def save_extrinsics(self):
        """Save current extrinsic parameters"""
        # In a real application, this would save to a file
        # For this demo, we'll just show a message
        
        extrinsics = {
            'translation': {
                'x': self.trans_x_var.get(),
                'y': self.trans_y_var.get(),
                'z': self.trans_z_var.get()
            },
            'rotation': {
                'x': self.rot_x_var.get(),
                'y': self.rot_y_var.get(),
                'z': self.rot_z_var.get(),
                'w': self.rot_w_var.get()
            }
        }
        
        # For demo purposes, print to console
        print("Saved extrinsics:", extrinsics)
        
        self.update_status("Extrinsic parameters saved")
        messagebox.showinfo("Parameters Saved", "Extrinsic parameters have been saved")
        
    def use_depth_values(self):
        """Use pixel coordinates and depth from the depth tab"""
        if self.pixel_x is not None and self.pixel_y is not None:
            self.coord_x_var.set(self.pixel_x)
            self.coord_y_var.set(self.pixel_y)
            
            if hasattr(self, 'depth_value') and self.depth_value is not None:
                # Convert from units to meters
                self.coord_depth_var.set(self.depth_value / 1000.0)
                
            self.update_status("Values from depth tab applied")
        else:
            self.update_status("No pixel coordinates available")
            messagebox.showinfo("No Data", "No pixel coordinates available. Please select a pixel in the Depth tab first.")
    
    def pixel_to_robot_base(self, pixel_x, pixel_y, depth):
        """Convert pixel coordinates to robot base coordinates"""
        # Extract intrinsic parameters
        fx = self.camera_intrinsics['K'][0]
        fy = self.camera_intrinsics['K'][4]
        cx = self.camera_intrinsics['K'][2]
        cy = self.camera_intrinsics['K'][5]
        
        # Build extrinsics from UI
        camera_extrinsics = {
            'translation': {
                'x': self.trans_x_var.get(),
                'y': self.trans_y_var.get(),
                'z': self.trans_z_var.get()
            },
            'rotation': {
                'x': self.rot_x_var.get(),
                'y': self.rot_y_var.get(),
                'z': self.rot_z_var.get(),
                'w': self.rot_w_var.get()
            }
        }
        
        # Extract extrinsic parameters
        translation = np.array([
            camera_extrinsics['translation']['x'],
            camera_extrinsics['translation']['y'],
            camera_extrinsics['translation']['z']
        ])
        
        quaternion = np.array([
            camera_extrinsics['rotation']['x'],
            camera_extrinsics['rotation']['y'],
            camera_extrinsics['rotation']['z'],
            camera_extrinsics['rotation']['w']
        ])
        
        # Convert pixel coordinates to normalized camera coordinates
        x_normalized = (pixel_x - cx) / fx
        y_normalized = (pixel_y - cy) / fy
        
        # Calculate 3D point in camera frame
        point_camera = np.array([
            x_normalized * depth,
            y_normalized * depth,
            depth
        ])
        
        # Convert quaternion to rotation matrix
        # scipy expects quaternion as [w, x, y, z], but our quaternion is [x, y, z, w]
        rot = R.from_quat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        rotation_matrix = rot.as_matrix()
        
        # Transform point from camera frame to robot base frame
        point_robot = rotation_matrix @ point_camera - translation
        
        # Apply final adjustments as in the original script
        point_robot[0] = (point_robot[0] + 0.02)
        point_robot[1] = -(point_robot[1] + 0.1)
        
        return point_robot
        
    def calculate_coordinates(self):
        """Calculate robot coordinates from pixel coordinates and depth"""
        try:
            # Get values from UI
            pixel_x = self.coord_x_var.get()
            pixel_y = self.coord_y_var.get()
            depth = self.coord_depth_var.get()
            
            # Calculate robot coordinates
            robot_coords = self.pixel_to_robot_base(pixel_x, pixel_y, depth)
            
            # Update display
            self.robot_x_var.set(f"{robot_coords[0]:.6f}")
            self.robot_y_var.set(f"{robot_coords[1]:.6f}")
            self.robot_z_var.set(f"{robot_coords[2]:.6f}")
            
            self.update_status(f"Calculated robot coordinates: {robot_coords}")
            
        except Exception as e:
            self.update_status(f"Error calculating coordinates: {str(e)}")
            messagebox.showerror("Calculation Error", f"Error calculating coordinates: {str(e)}")
            
    def copy_coordinates(self):
        """Copy calculated coordinates to clipboard"""
        if self.robot_x_var.get() != "Not calculated":
            coords_text = f"X: {self.robot_x_var.get()}, Y: {self.robot_y_var.get()}, Z: {self.robot_z_var.get()}"
            self.root.clipboard_clear()
            self.root.clipboard_append(coords_text)
            self.update_status("Coordinates copied to clipboard")
        
def main():
    root = tk.Tk()
    app = IntegratedApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()