#!/usr/bin/env python3

import sys
import os
import rospy
import numpy as np
from math import pi
import tkinter as tk
from tkinter import ttk, messagebox
from ur3e_hardware_controller import UR3eRealRobotController

class UR3eGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UR3e Robot Control")
        self.root.geometry("800x680")  # Increased height to accommodate the save position frame
        
        # Initialize ROS controller in a separate try-except block
        try:
            self.controller = None  # Initialize to None first
            self.init_status_label = ttk.Label(self.root, text="Initializing robot controller...", foreground="blue")
            self.init_status_label.pack(pady=10)
            self.root.update()  # Update the UI to show the initializing message
            
            # Initialize the controller
            self.controller = UR3eRealRobotController()
            self.init_status_label.config(text="Robot controller initialized successfully!", foreground="green")
        except Exception as e:
            self.init_status_label.config(text=f"Failed to initialize controller: {str(e)}", foreground="red")
            messagebox.showerror("Initialization Error", f"Failed to initialize robot controller: {str(e)}")

        # Create main frame
        main_frame = ttk.Frame(root)
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
        self.velocity_var = tk.DoubleVar(value=0.1)  # Default 10%
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
            
            # Confirm movement
            if not messagebox.askyesno("Confirm Movement", 
                                     f"Move robot to:\nq1={np.rad2deg(joint_values[0]):.2f}°, "
                                     f"q2={np.rad2deg(joint_values[1]):.2f}°, "
                                     f"q3={np.rad2deg(joint_values[2]):.2f}°, "
                                     f"q4={np.rad2deg(joint_values[3]):.2f}°, "
                                     f"q5={np.rad2deg(joint_values[4]):.2f}°, "
                                     f"q6={np.rad2deg(joint_values[5]):.2f}°\n"
                                     f"with velocity scaling: {velocity:.2f}?"):
                return
            
            # Update status
            self.status_var.set(f"Moving robot to joint position... (velocity: {velocity:.2f})")
            self.root.update()  # Force UI update
            
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
            self.root.update()  # Force UI update
            
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
            
            # Set position name in the save field if it's a custom position
            if "Pick up" in name or "Drop-off" in name:
                # Extract just the position name without the type prefix
                if "Pick up" in name:
                    self.position_type_var.set("Pick up")
                    base_name = name.replace("Pick up ", "")
                elif "Drop-off" in name:
                    self.position_type_var.set("Drop-off")
                    base_name = name.replace("Drop-off ", "")
                
                # Set the position name
                if base_name.isdigit():  # If it's just a number
                    self.position_name_var.set(f"pos {base_name}")
                else:
                    self.position_name_var.set(base_name)
        
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

def main():
    # Initialize Tkinter
    root = tk.Tk()
    
    # Enable high DPI scaling on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # Create application
    app = UR3eGUI(root)
    
    # Start main loop
    try:
        root.mainloop()
    finally:
        # Clean shutdown of ROS when closing the GUI
        if hasattr(app, 'controller') and app.controller is not None:
            try:
                rospy.signal_shutdown("GUI closed")
                print("ROS shutdown initiated.")
            except:
                pass

if __name__ == "__main__":
    main()