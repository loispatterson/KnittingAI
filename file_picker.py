import tkinter as tk
from tkinter import filedialog

def select_image_file(title="Select an image for prediction"):
    """
    Opens a system file dialog to choose an image file.
    Returns the selected file path as a string, or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
    )
    root.destroy()
    return file_path or None