# Real-Time Object Detection with YOLOv5 and OpenCV

This Python application captures live video from a webcam, processes it to detect specific objects (bottle, fork, book), and summarizes the video statistics in real time. The application uses the YOLOv5 model for object detection and OpenCV for video capture and processing.

## Features

- **Real-Time Object Detection**: Detects objects as the video streams from the webcam.
- **Video Statistics Summary**: Outputs the total number of frames processed, total minutes of video, and the count of frames where each specified object was detected.
- **Performance Metrics**: Displays the frames per second (FPS) processed, providing insights into the application's performance.

## Requirements

To run this application, you will need:

- Python 3.8 or later
- OpenCV
- PyTorch
- Torchvision

## Installation

First, clone this repository:

```bash
git clone https://github.com/sivkaur/Real-Time-Object-Detection.git
```

Install the required packages:
```bash
pip install opencv-python torch torchvision
```

## Usage
To start the application, run the following command in your terminal:

```bash
python3 object_detection.py
```

Make sure your webcam is enabled and properly configured on your machine.

## Code Overview
The main components of the application are:

#### Video Capture Initialization

- The code initializes a video capture object `cap` using OpenCV to capture video from the webcam.

#### Preprocessing Functions

- **equalize_histogram_color**: Applies histogram equalization to the Y channel of the frame converted to the YCrCb color space. This step enhances the contrast, making the detection process more robust against different lighting conditions.
- **reduce_noise**: Implements Gaussian Blur to smooth the frame, reducing noise that could lead to false positives in detection.

#### Object Detection

- The code loads the YOLOv5 model from `torch.hub`, which is an efficient way to get pre-trained models. 
- Processes each frame through the model after converting it from BGR to RGB and adjusting dimensions.

#### Results Rendering and Display

- Detected objects are handled within a loop where each detection above the confidence threshold is marked, annotated, and counted. This part uses the model’s class names to check if detected objects belong to the categories of interest ('bottle', 'fork', 'book').
- The rectangles and labels are drawn and placed on the frames, ensuring that detections are visually marked in the output video.

#### FPS Calculation and Display

- FPS is calculated using the time at the start and the end of the frame processing, which provides feedback on the system's performance. Displaying FPS on the frame using `cv2.putText` right before showing the frame ensures it is updated and visible in real time.

#### Clean-up and Statistics

- The loop breaks upon pressing 'q', and the clean-up code properly releases the webcam and destroys all OpenCV windows. Final statistics about the session are printed to give insights into the overall performance and activity during the run.
