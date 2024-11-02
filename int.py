import B_1
import tkinter as tk
from tkinter import messagebox
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Forming Conditions")
root.geometry("1200x600")

# Lists to store boundary and interface conditions
boundary_conditions = []
boundary_conditions_entries = []

# Lists for entry widgets for easy access
boundary_condition_widgets = []
interface_condition_widgets = []

# Function to clear existing condition entry widgets
def clear_condition_widgets():
    for widget in boundary_condition_widgets + interface_condition_widgets:
        widget.destroy()
    boundary_condition_widgets.clear()
    interface_condition_widgets.clear()

# Function to get values and display results
def get_values():
    try:
        x1_min = float(x1_min_entry.get())
        x2_min = float(x2_min_entry.get())
        width = float(width_entry.get())
        height = float(height_entry.get())
        T = float(T_entry.get())
        t_graph = float(t_entry.get())
        x1_max = x1_min + height
        x2_max = x2_min + width

        results = f"x1 range: {x1_min} to {x1_max}\n"
        results += f"x2 range: {x2_min} to {x2_max}\n"
        results += f"T value: {T}\n"
        results += f"t value for graph: {t_graph}\n"
        results += "Selected function: x1 ** 2 + x2 ** 2 + T\n"
        results += "Defined function u(s): 5\n"
        results += f"Boundary conditions: {boundary_conditions}\n"
        results += f"Interface conditions: {boundary_conditions_entries}"
        messagebox.showinfo("Results", results)

        # Create 3D graph in a new window
        s = B_1.Solver()
        s.lbr = [x1_min, x2_min]
        s.height = height
        s.width = width
        s.T = T
        s.t_inst = t_graph
        s.Set_L_0_r = boundary_conditions
        s.Set_L_G_r = boundary_conditions_entries
        s.shov_y_primary()
        s.main()
        
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Function to dynamically enter conditions
operator_options = ["1", "похідна по t", "похідна по x1"]
operator_numbers = [1, 2, 3]  # Numeric representations
def enter_conditions():
    try:
        # Clear existing condition entry fields
        clear_condition_widgets()

        # Get the number of boundary and interface conditions
        num_boundary_conditions = int(num_boundary_conditions_entry.get())
        num_interface_conditions = int(num_interface_conditions_entry.get())
        
        if num_boundary_conditions <= 0 or num_interface_conditions <= 0:
            raise ValueError("The number of conditions must be positive.")

        # Create entry fields for each boundary condition
        for i in range(num_boundary_conditions):
            tk.Label(root, text=f"Boundary Condition {i + 1}:").grid(row=10 + i, column=0, sticky=tk.W)
            x1_entry = tk.Entry(root, width=10)
            x2_entry = tk.Entry(root, width=10)
            t_entry = tk.Entry(root, width=10)

            # Create OptionMenu for operator
            operator_var = tk.StringVar(value=operator_options[0])
            operator_menu = tk.OptionMenu(root, operator_var, *operator_options)

            x1_entry.grid(row=10 + i, column=1)
            x2_entry.grid(row=10 + i, column=2)
            t_entry.grid(row=10 + i, column=3)
            operator_menu.grid(row=10 + i, column=4)

            # Store widgets
            boundary_condition_widgets.extend([x1_entry, x2_entry, t_entry, operator_menu])

        # Create entry fields for each interface condition
        for i in range(num_interface_conditions):
            tk.Label(root, text=f"Interface Condition {i + 1}:").grid(row=10 + num_boundary_conditions + i, column=0, sticky=tk.W)
            x1_entry = tk.Entry(root, width=10)
            x2_entry = tk.Entry(root, width=10)
            t_entry = tk.Entry(root, width=10)

            # Create OptionMenu for operator
            operator_var = tk.StringVar(value=operator_options[0])
            operator_menu = tk.OptionMenu(root, operator_var, *operator_options)

            x1_entry.grid(row=10 + num_boundary_conditions + i, column=1)
            x2_entry.grid(row=10 + num_boundary_conditions + i, column=2)
            t_entry.grid(row=10 + num_boundary_conditions + i, column=3)
            operator_menu.grid(row=10 + num_boundary_conditions + i, column=4)

            # Store widgets
            interface_condition_widgets.extend([x1_entry, x2_entry, t_entry, operator_menu])

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

# Function to save boundary conditions
def save_boundary_conditions():
    try:
        # Clear existing boundary conditions to avoid duplicates
        boundary_conditions.clear()

        # Loop through all boundary condition widgets
        for i in range(len(boundary_condition_widgets) // 4):
            # Get the corresponding entries and operator for each condition
            x1_entry = boundary_condition_widgets[i * 4]
            x2_entry = boundary_condition_widgets[i * 4 + 1]
            t_entry = boundary_condition_widgets[i * 4 + 2]
            operator_var = boundary_condition_widgets[i * 4 + 3].cget("text")

            # Convert the entries to float
            x1 = float(x1_entry.get())
            x2 = float(x2_entry.get())
            t = float(t_entry.get())

            # Find the numeric operator value
            operator_num = operator_numbers[operator_options.index(operator_var)]  # Get the numeric value
            
            # Append to the boundary_conditions
            boundary_conditions.append([x1, x2, t, operator_num])

        # Show message with all saved conditions
        messagebox.showinfo("Success", f"All boundary conditions saved: {boundary_conditions}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all boundary conditions.")

# Function to save interface conditions with selected operators
def save_interface_conditions():
    try:
        # Clear existing entries to avoid duplicates
        boundary_conditions_entries.clear()

        # Loop through all interface condition widgets
        for i in range(len(interface_condition_widgets) // 4):
            # Get the corresponding entries and operator for each condition
            x1_entry = interface_condition_widgets[i * 4]
            x2_entry = interface_condition_widgets[i * 4 + 1]
            t_entry = interface_condition_widgets[i * 4 + 2]
            operator_var = interface_condition_widgets[i * 4 + 3].cget("text")

            # Convert the entries to float
            x1 = float(x1_entry.get())
            x2 = float(x2_entry.get())
            t = float(t_entry.get())

            # Find the numeric operator value
            operator_num = operator_numbers[operator_options.index(operator_var)]  # Get the numeric value
            
            # Append to the boundary_conditions_entries
            boundary_conditions_entries.append([x1, x2, t, operator_num])

        # Show message with all saved conditions
        messagebox.showinfo("Success", f"All interface conditions saved: {boundary_conditions_entries}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all interface conditions.")

# Function to clear all conditions
def save_all_conditions():
    save_boundary_conditions()
    save_interface_conditions()

# Function to clear all boundary and interface conditions
def clear_conditions():
    # Clear stored boundary and interface conditions
    boundary_conditions.clear()
    boundary_conditions_entries.clear()

# Button to clear all conditions
clear_conditions_button = tk.Button(root, text="Clear All Conditions", command=clear_conditions)
clear_conditions_button.grid(row=8, column=8, columnspan=6)

# Display selected and defined functions
tk.Label(root, text="Selected function: x1 ** 2 + x2 ** 2 + T; u(s): 5; v: [0,0]; k = 1").grid(row=0, column=4)

# UI Elements for input
tk.Label(root, text="Minimum value of x1:").grid(row=0, column=0)
x1_min_entry = tk.Entry(root, width=10)
x1_min_entry.grid(row=0, column=1)
x1_min_entry.insert(0, "0")

tk.Label(root, text="Minimum value of x2:").grid(row=3, column=0)
x2_min_entry = tk.Entry(root, width=10)
x2_min_entry.grid(row=3, column=1)
x2_min_entry.insert(0, "0")

tk.Label(root, text="Width (for x1):").grid(row=4, column=0)
width_entry = tk.Entry(root, width=10)
width_entry.grid(row=4, column=1)
width_entry.insert(0, "2")

tk.Label(root, text="Height (for x2):").grid(row=5, column=0)
height_entry = tk.Entry(root, width=10)
height_entry.grid(row=5, column=1)
height_entry.insert(0, "2")

# Entry for T with default value 0
tk.Label(root, text="Value of T:").grid(row=6, column=0)
T_entry = tk.Entry(root, width=10)
T_entry.grid(row=6, column=1)
T_entry.insert(0, "2")

# Entry for t value for graph with default value 0
tk.Label(root, text="t value for graph:").grid(row=7, column=0)
t_entry = tk.Entry(root, width=10)
t_entry.grid(row=7, column=1)
t_entry.insert(0, "0")

# Fields for entering the number of boundary and interface conditions
tk.Label(root, text="Number of boundary conditions:").grid(row=8, column=0)
num_boundary_conditions_entry = tk.Entry(root, width=10)
num_boundary_conditions_entry.grid(row=8, column=1)
num_boundary_conditions_entry.insert(0, "1")

tk.Label(root, text="Number of interface conditions:").grid(row=8, column=3)
num_interface_conditions_entry = tk.Entry(root, width=10)
num_interface_conditions_entry.grid(row=8, column=4)
num_interface_conditions_entry.insert(0, "1")

# Buttons to save conditions
get_values_button = tk.Button(root, text="Get Values", command=get_values)
get_values_button.grid(row=0, column=2)

enter_conditions_button = tk.Button(root, text="Enter Conditions", command=enter_conditions)
enter_conditions_button.grid(row=8, column=5)

save_all_conditions_button = tk.Button(root, text="Save All Conditions", command=save_all_conditions)
save_all_conditions_button.grid(row=9, column=0, columnspan=6)

root.mainloop()
