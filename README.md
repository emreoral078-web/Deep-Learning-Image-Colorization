# Histogram-Guided Fusion U-Net for Image Colorization

This project was developed as the final project for the **CS464: Introduction to Machine Learning** course at Bilkent University. It explores advanced deep learning architectures to solve the "ill-posed" problem of grayscale image colorization.

##  Team (Group 20)
* **Yılmaz Öncü**
* **Ozan Sazak**
* **Mustafa Boran Kaya**
* **Emre Oral**
* **Umut Yalçındağ**

##  Project Overview
Traditional colorization models often suffer from the **"Sepia Effect"**—a phenomenon where the model predicts the statistical mean of all possible colors, resulting in desaturated and brownish outputs. 

Our work introduces a **Histogram-Guided Fusion U-Net** that:
1. Uses a **U-Net architecture** to preserve high-frequency spatial details.
2. Incorporates **Global Color Priors** via a Histogram Fusion layer to allow controllable colorization.
3. Optimizes using **Mean Absolute Error (MAE)** loss instead of MSE to encourage more vibrant and diverse color predictions.

##  Architecture
The model consists of two main branches:
* **Spatial Branch (Encoder):** A series of double-convolution blocks that extract structural features from the $160 \times 160$ grayscale input.
* **Global Branch (MLP):** A Multi-Layer Perceptron that processes a 96-bin color histogram to provide style guidance.
* **Fusion Layer:** A concatenation mechanism that injects the global color features into the bottleneck of the U-Net.

##  Key Features
* **Controllability:** By changing the input histogram, users can manipulate the "mood" or color distribution of the output.
* **Saturation Boosting:** Includes a post-processing step to align the model's output with human visual perception.
* **Evaluation:** Benchmarked against a Baseline CNN and a standard U-Net using Qualitative Analysis and Loss Curves.

##  Repository Structure
* `CS464_Team20_Code.py`: Full implementation using TensorFlow/Keras, including data pipeline (Intel Image Dataset), model architectures, and training loops.
* `CS464_Team20_FinalProjectReport.pdf`: Detailed academic report covering the methodology, theoretical background, and comprehensive results.
* `README.md`: Project documentation.

##  Requirements
* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

##  Results
Our Fusion model successfully disentangles content from style, effectively breaking the "regression-to-the-mean" trap found in simpler architectures. The use of MAE loss significantly reduced the desaturation issues common in MSE-based models.
