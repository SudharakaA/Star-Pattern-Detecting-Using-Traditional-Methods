import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
import numpy as np
import os

# Initialize variables
original_img = None
img = None
img_display = None
patterns = []

# Functionality for the buttons
def load_image():
    global img, img_display, original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        original_img = Image.open(file_path)
        original_img = original_img.resize((600, 600))  # Resize to 600x600
        img = original_img.copy()

        # Check if image is a night sky before updating canvas
        if is_night_sky(img):
            result_label.config(text="Night sky detected!")
            update_canvas(img)
        else:
            result_label.config(text="Not a night sky image. Please upload a night sky image.")

def update_canvas(image):
    global img_display
    img_display = ImageTk.PhotoImage(image)
    canvas.config(width=600, height=600)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_display)
    canvas.image = img_display

def is_night_sky(image):
    """Check if the image is a night sky by assessing brightness and detecting star-like features."""
    # Convert PIL image to grayscale OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Step 1: Brightness Check
    brightness = np.mean(img_cv)
    if brightness > 70:  # Threshold for brightness, adjust based on sample images
        return False
    
    # Step 2: Detect Star-like Features
    # Use thresholding to find bright spots, which may indicate stars
    _, thresh_img = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY)  # Threshold for star detection
    white_pixels = cv2.countNonZero(thresh_img)
    
    # Check if there's a significant number of "star-like" pixels (small bright points)
    if white_pixels < 50:  # Threshold for the minimum number of star-like spots
        return False
    
    return True

# Additional processing functions remain the same
def convert_to_grayscale():
    global img
    if img:
        img = ImageOps.grayscale(img)
        update_canvas(img)

def sharpen_image():
    global img
    if img:
        laplacian_kernel = [
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        ]
        img = img.filter(ImageFilter.Kernel((3, 3), laplacian_kernel, 1, 0))
        update_canvas(img)

def remove_salt_pepper():
    global img
    if img:
        img_cv = np.array(img.convert('L'))
        img_cv = cv2.medianBlur(img_cv, 3)
        img = Image.fromarray(img_cv)
        update_canvas(img)

def reset_filters():
    global img
    if original_img:
        img = original_img.copy()
        update_canvas(img)

def rotate_image():
    global img
    if img:
        img = img.rotate(90, expand=True)
        update_canvas(img)

def load_patterns():
    pattern_dir = "patterns/"
    for filename in os.listdir(pattern_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            pattern_img = cv2.imread(os.path.join(pattern_dir, filename), 0)
            patterns.append((pattern_img, filename.split('.')[0]))

def find_pattern():
    global img
    if img:
        img_cv = np.array(img.convert('L'))
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_cv, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        best_match_name = None
        best_match_score = float('inf')

        for pattern_img, pattern_name in patterns:
            kp2, des2 = orb.detectAndCompute(pattern_img, None)
            if des2 is not None:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                match_score = sum([match.distance for match in matches[:10]])
                if match_score < best_match_score:
                    best_match_score = match_score
                    best_match_name = pattern_name

        if best_match_name:
            result_label.config(text=f"Best Match: {best_match_name}")
        else:
            result_label.config(text="No match found")

# Create the main window
root = tk.Tk()
root.title("Night Sky Image Preprocessing App")
root.geometry("600x650")

canvas = tk.Canvas(root, width=600, height=600)
canvas.grid(row=0, column=0, rowspan=7)

load_patterns()

load_btn = tk.Button(root, text="Load Image", command=load_image)
load_btn.grid(row=0, column=1)

grayscale_btn = tk.Button(root, text="Convert to Grayscale", command=convert_to_grayscale)
grayscale_btn.grid(row=1, column=1)

sharpen_btn = tk.Button(root, text="Sharpen Image", command=sharpen_image)
sharpen_btn.grid(row=2, column=1)

remove_noise_btn = tk.Button(root, text="Remove Salt & Pepper", command=remove_salt_pepper)
remove_noise_btn.grid(row=3, column=1)

rotate_btn = tk.Button(root, text="Rotate 90°", command=rotate_image)
rotate_btn.grid(row=4, column=1)

find_btn = tk.Button(root, text="Find Pattern", command=find_pattern)
find_btn.grid(row=5, column=1)

reset_btn = tk.Button(root, text="Remove All Filters", command=reset_filters)
reset_btn.grid(row=6, column=1)

result_label = tk.Label(root, text="No match found")
result_label.grid(row=7, column=0, columnspan=2)

root.mainloop()
