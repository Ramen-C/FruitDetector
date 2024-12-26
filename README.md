

**FruitDetector** is a Python-based Graphical User Interface (GUI) application that leverages OpenCV for image processing to automatically detect and count apples, bananas, and oranges within an image. Utilizing techniques such as color space conversion, thresholding, morphological operations, and the watershed algorithm, the application accurately identifies and classifies fruits, providing performance evaluation metrics based on ground truth data.

## About the Project

In the realm of fruit classification and counting, accuracy and efficiency are paramount. **FruitDetector** offers a user-friendly interface that allows users to effortlessly load images, perform fruit detection, and observe each stage of the image processing pipeline. This project is suitable for research, educational purposes, and can be extended for practical applications in agriculture, retail, and quality control.

## Features

- **Multi-Scenario Support**: Supports multiple scenarios (e.g., ScenarioA, ScenarioB) each containing different sets of fruit images.
- **Image Loading and Display**: Users can select different scenarios and images, with the application displaying the original and processed images at various stages.
- **Fruit Detection and Counting**: Utilizes color space conversion, thresholding, morphological operations, and watershed algorithm to detect and count apples, bananas, and oranges in images.
- **Performance Evaluation**: Calculates Precision, Recall, and F1 Score based on ground truth data to assess the accuracy of fruit detection.
- **Dynamic Image Resizing**: Automatically adjusts image sizes based on window dimensions while maintaining aspect ratios.
- **Debugging Information**: Outputs mask pixel counts and detected fruit counts to the console for development and debugging purposes.

## Tech Stack

- **Programming Language**: Python 3.x
- **GUI Framework**: Tkinter
- **Image Processing**: OpenCV, PIL (Pillow)
- **Data Handling**: NumPy
