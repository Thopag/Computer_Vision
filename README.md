
# Overview
This project is developed as part of a Computer Vision course.
It uses several classical and well-known computer vision algorithms, including:

- Kalman Filter for object tracking and trajectory smoothing

- Canny Edge Detection for edge extraction

- Morphological Operations such as dilation and erosion

The goal of the project is to detect and track objects over time in video sequences, preprocess their trajectories, and then use this information offline to carefully generate the desired final effects.

# Project Structure

The workflow is divided into two main stages:

-  Preprocessing Stage
        During preprocessing:
        Multiple videos are analyzed ,Several .txt files are generated containing object positions over time
        Tracking data is refined using:
        Smoothing
        Interpolation
        Noise reduction
        These processed files are saved and reused later, avoiding real-time constraints.
        
-  Offline Processing Stage
        The preprocessed data is then used offline to:        Accurately reconstruct object motion        Generate the final visual effects (“tricks”) in a controlled and precise manner

        files
        │
        ├── color_output        # Stores color masks
        ├── interpolation       # Stores interpolated trajectory files
        ├── object_&_wand       # Stores object and wand detection results
        ├── object_tracking     # Stores object tracking data
        ├── wand_tracking       # Stores wand tracking data
        └── yolo_output         # Stores YOLO detection outputs
