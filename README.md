# Document-Scanner-Using-Python-and-OpenCV

This project demonstrates various image processing and computer vision techniques using OpenCV. The main goal is to apply multiple transformations to images, including object detection, feature extraction, image enhancement, and motion detection.

## Features

- **Image Enhancement**: Applies techniques such as histogram equalization, smoothing, and sharpening to enhance image quality.
- **Object Detection**: Implements object detection algorithms using pre-trained classifiers (e.g., Haar cascades) to detect faces, eyes, or other objects in images.
- **Edge Detection**: Uses algorithms like Canny Edge Detection to identify edges in images.
- **Motion Detection**: Detects motion in video streams by analyzing frame differences and identifying changes.
- **Feature Matching**: Uses techniques like SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) to detect and match features between images.
- **Contours Detection**: Identifies and marks contours in images, useful for shape detection.
- **Real-time Processing**: Capable of processing real-time video streams for applications like object tracking.

## File Structure

- `opencv_project.py`: Main Python script where all OpenCV functionalities (object detection, feature extraction, etc.) are implemented.
- `images/`: Directory containing sample images for testing the project.
- `output/`: Directory where processed images and results (e.g., detections, enhancements) are saved.
- `README.md`: This file containing the project documentation.

## Functions

### 1. **`image_enhancement`**
   - **Description**: Enhances the quality of the input image using histogram equalization, smoothing, and sharpening techniques.
   - **Input**: `image` (Original image).
   - **Output**: Returns the enhanced image.

### 2. **`detect_objects`**
   - **Description**: Detects faces, eyes, or other objects in the input image using pre-trained Haar cascade classifiers.
   - **Input**: `image` (Input image).
   - **Output**: Returns the image with detected objects outlined or marked.

### 3. **`edge_detection`**
   - **Description**: Applies the Canny edge detection algorithm to identify edges in the image.
   - **Input**: `image` (Input image).
   - **Output**: Returns the image with detected edges.

### 4. **`motion_detection`**
   - **Description**: Detects motion in a video stream by analyzing differences between consecutive frames.
   - **Input**: `video` (Input video stream).
   - **Output**: Displays the motion detected in real-time.

### 5. **`feature_matching`**
   - **Description**: Detects and matches features between two images using SIFT or ORB algorithms.
   - **Input**: `image1`, `image2` (Two input images to compare).
   - **Output**: Returns the matched features between the two images.

### 6. **`contour_detection`**
   - **Description**: Detects and marks contours in the input image.
   - **Input**: `image` (Input image).
   - **Output**: Returns the image with detected contours.

## Requirements

- **Python 3.x**
- **OpenCV**
- **NumPy**
