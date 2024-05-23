# Face Detection using Viola-Jones Algorithm

## Project Overview
This project aims to detect faces, eyes, and smiles using the Viola-Jones Algorithm. The Viola-Jones algorithm is a popular object detection framework used for real-time face detection. It is based on machine learning techniques and uses Haar-like features to identify objects within images or videos efficiently. This project utilizes OpenCV for image processing and Streamlit for building an interactive web application.

## Key Features
- Detect faces, eyes, and smiles in images or videos.
- Real-time detection using webcam.
- Interactive web application using Streamlit.

## Libraries Used
- **cv2**: OpenCV library for image and video processing.
- **Streamlit**: Library for building interactive web applications with Python.

## Viola-Jones Algorithm
The Viola-Jones algorithm is a machine learning algorithm used for object detection, particularly for detecting human faces in images or videos. It uses a trained model based on Haar-like features to identify regions in an image that likely contain a face. The algorithm is known for its accuracy and speed, making it popular for real-time applications.

## How It Works
1. **Loading the Face Cascade Classifier**: The face cascade classifier is a pre-trained model used to detect faces. It uses Haar-like features to identify regions in an image that likely contain a face.
2. **Detection Process**:
   - **Face Detection**: Identify and draw bounding boxes around faces in the image.
   - **Eyes Detection**: Identify and draw bounding boxes around eyes within the detected faces.
   - **Smile Detection**: Identify and draw bounding boxes around smiles within the detected faces.


## How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install opencv-python-headless streamlit
   ```
2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Conclusion
This project demonstrates the implementation of the Viola-Jones algorithm for real-time face detection using OpenCV and Streamlit. The interactive web application allows users to upload images and visualize the detection results, making it a useful tool for various applications in image processing and computer vision.

## Acknowledgements
- [OpenCV Documentation](https://opencv.org/)
- [Streamlit Documentation](https://streamlit.io/)
- [Viola-Jones Algorithm](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)

## License
This project is licensed under the MIT License.
