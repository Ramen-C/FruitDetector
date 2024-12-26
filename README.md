

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

---

本项目是基于数字图像处理（DIP）的基础上开发的，图像识别部分仅依赖HSV颜色区分和形态学完成，没有进行深度学习。
识别水果只有香蕉、苹果、橘子。
处理如下：

-图像读入与预处理：
概念：利用数字图像处理技术，将原始RGB图像转换至适合颜色分析的HSV空间，同时对图像进行平滑滤波（如高斯滤波）以降低噪声。

-阈值分割与二值化：
概念：根据事先定义的颜色范围（通过实验调整阈值）对图像进行阈值分割，使目标水果区域在二值图中清晰凸显。

-形态学处理：
概念：对二值图像采用腐蚀和膨胀等基本形态学操作，以消除离散噪声点与填补小孔洞。使得水果轮廓更加闭合，从而为后续轮廓提取与连通域分析提供良好基础。

-连通域标记与特征提取：
概念：对形态学处理后的图像开始连通域标记操作，以区分不同的前景物体。随后，对每个连通区域计算基本特征（面积、周长、颜色均值），实现简单的类型判定。

-粘连目标的分割（分水岭算法）：
概念：当检测到有较大连通域内包含多个粘连在一起的水果时，需利用分水岭(Watershed)算法将其分离。

原理：
1. 首先计算距离变换，以度量前景点到背景的距离分布。
2. 对距离图中局部极值点进行标记，这些标记被作为分水岭的源。
3. 利用图像梯度信息通过分水岭算法在山脊处对前景区域进行分割，将原本相互粘连的水果分离成多个独立个体。
