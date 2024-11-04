# Night Sky Image Preprocessing App 🌌

A Python-based desktop application using Tkinter and OpenCV for preprocessing night sky images and identifying predefined star patterns. This tool allows users to load images, apply filters, detect night sky features, and search for specific star patterns. The application is helpful for amateur astronomers and those working on projects involving star pattern recognition.

## Features

- **Load Images**: Import night sky images to begin preprocessing.
- **Night Sky Detection**: Analyzes the brightness and star-like features to confirm the image contains a night sky.
- **Image Filters**:
  - Convert to Grayscale
  - Sharpen (using Laplacian kernel)
  - Remove Salt & Pepper Noise
  - Rotate Image
- **Pattern Matching**: Matches loaded images with predefined star patterns to identify known star formations.
- **Reset Filters**: Allows users to remove all applied filters and revert to the original image.

## Technologies & Libraries Used

- **Python**: Core programming language for the application.
- **Tkinter**: For creating the graphical user interface.
- **PIL (Pillow)**: For image manipulation and display.
- **OpenCV**: For image processing tasks, such as detecting star-like features and matching patterns.
- **NumPy**: For numerical operations and array manipulation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/night-sky-preprocessing-app.git
   cd night-sky-preprocessing-app
2. **Install the required dependencies: Make sure you have Python installed (version 3.7+ is recommended).
   pip install -r requirements.txt

3. If you don't have a requirements.txt file, you can install dependencies individually:
   pip install opencv-python-headless numpy pillow
