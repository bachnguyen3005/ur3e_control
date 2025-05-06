import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import os
import sys
from PIL import Image, ImageTk
import argparse

# Import the shape detection functions
from shape_color_detection import detect_shapes_and_colors, detect_shapes_and_colors_yolo, detect_shapes_and_colors_yolo_seg

class ShapeDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape and Color Detection")
        self.root.geometry("1200x800")
        
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
        self.yolo_model_path = "/home/dinh/catkin_ws/src/ur3e_control/scripts/segment.pt" 
        
        # Create frames
        self.create_frames()
        
        # Create widgets
        self.create_widgets()
        
        # Initialize status
        self.update_status("Ready. Please load an image.")
    
    def create_frames(self):
        # Main frame layout
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding="10", width=200)
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
        self.status_frame = ttk.Frame(self.root, padding="5")
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
        
        # Detect buttons section
        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Traditional detection button
        self.detect_button = ttk.Button(
            self.right_frame, 
            text="Detect Shapes with OpenCV", 
            command=self.detect_shapes,
            state=tk.DISABLED
        )
        self.detect_button.pack(fill=tk.X, pady=5)
        
        # YOLO detection button
        self.detect_yolo_button = ttk.Button(
            self.right_frame, 
            text="Detect Shapes with YOLO", 
            command=self.detect_shapes_yolo,
            state=tk.DISABLED
        )
        self.detect_yolo_button.pack(fill=tk.X, pady=5)
        
        # Display model path (informational only)
        model_info = f"YOLO Model: {os.path.basename(self.yolo_model_path)}"
        ttk.Label(self.right_frame, text=model_info, foreground="blue").pack(anchor=tk.W, pady=(5,10))
        
        # Reset view button
        ttk.Button(self.right_frame, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.update_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        
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
                self.show_image(self.display_image)
                
                self.update_status(f"Loaded image: {os.path.basename(file_path)}")
                self.detect_button.config(state=tk.NORMAL)
                self.detect_yolo_button.config(state=tk.NORMAL)
                
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
        """Run shape detection on the selected ROI or the entire image,
        ensuring that all coordinates are displayed relative to the whole image."""
        if self.original_image is None:
            self.update_status("Please load an image first")
            return
        
        try:
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
                # Pass ROI offset to ensure coordinates are global
                cropped_result, self.detected_squares = detect_shapes_and_colors(cropped, target, roi_offset=(start_x, start_y))
                
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
                            

                            
            else:
                # Process the entire image - no ROI offset needed
                self.result_image, self.detected_squares = detect_shapes_and_colors(image_to_process, target)
                
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.show_image(display_result)
            
            self.update_status(f"Detection completed for target: {target}")
            
        except Exception as e:
            self.update_status(f"Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def detect_shapes_yolo(self):
        """Run YOLO-based shape detection on the image"""
        if self.original_image is None:
            self.update_status("Please load an image first")
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
                cropped_result, self.detected_squares = detect_shapes_and_colors_yolo_seg(
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
                self.result_image, self.detected_squares, self.detected_circle = detect_shapes_and_colors_yolo_seg(
                    image_to_process, 
                    target, 
                    model_path=self.yolo_model_path
                )
                
                # Add YOLO model info
                h, w = self.result_image.shape[:2]
                model_name = os.path.basename(self.yolo_model_path)
                cv2.putText(self.result_image, f"YOLO Model: {model_name}", (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Convert result to RGB for display
            display_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # Show the result
            self.show_image(display_result)
            
            # Report detected squares
            if self.detected_squares:
                square_info_parts = []
                for square in self.detected_squares:
                    if len(square) == 4:  # Format with rotation
                        x, y, color, rotation = square
                        square_info_parts.append(f"{color} at ({x},{y}), rotation: {rotation:.1f}")
                    else:  # Original format without rotation
                        x, y, color = square
                        square_info_parts.append(f"{color} at ({x},{y})")
                square_info = ", ".join(square_info_parts)
                self.update_status(f"YOLO detection completed. Found squares: {square_info}")
            else:
                self.update_status(f"YOLO detection completed for target: {target}. No squares found.")
            
            if self.detected_circle:
                circle_info = ", ".join([f"{color} at ({x},{y})" for x, y, color in self.detected_circle])
                self.update_status(f"YOLO detection completed. Found circle: {circle_info}")
            
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

def main():
    # Create the main window
    root = tk.Tk()
    app = ShapeDetectionGUI(root)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()