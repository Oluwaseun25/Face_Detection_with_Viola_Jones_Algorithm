# # # # #import libraries 

# # # # import cv2
# # # # import streamlit as st 

# # # # #Load the face cascade classifier
# # # # #The code loads the face cascade classifier file from the specified path.
# # # # #facial detection
# # # # face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # # # face_cascade = cv2.CascadeClassifier(face_cascade_name)

# # # # # eye detection
# # # # eye_cascade_name = cv2.data.haarcascades + 'haarcascade_eye.xml'
# # # # eye_cascade = cv2.CascadeClassifier(eye_cascade_name)

# # # # # Smile detection
# # # # smile_cascade_name = cv2.data.haarcascades + 'haarcascade_smile.xml'
# # # # smile_cascade = cv2.CascadeClassifier(smile_cascade_name)
# # # # #Create a function to capture frames from the webcam and detect faces

# # # # def detect_faces():
# # # #     # Initialize the webcam #access the webcam (camera) connected to your computer
# # # #     cap = cv2.VideoCapture(0) #0 refers to the default camera connected to your system.
# # # #     while True:

# # # #         # Read the frames from the webcam
# # # #         ret, frame = cap.read()
# # # #         #It will be True if a frame is read successfully and False if there are no more frames to read.
# # # #         if frame is not None:
# # # #              # Convert the frames to grayscale
# # # #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # #             # Further processing of the grayscale image
# # # #         else:
# # # #             print("Error: Frame is empty")
       

# # # #         # Detect the faces using the face cascade classifier
# # # #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# # # #         # Draw rectangles around the detected faces
# # # #         #Rectangles are drawn around each detected face in the frame image to visually indicate the location of the detected faces.
# # # #         for (x, y, w, h) in faces:
# # # #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
# # # #         # Display the frames
# # # #         st.image(frame, channels="BGR")

# # # #         # Exit the loop when 'q' is pressed
# # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # #             break

# # # #     # Release the webcam and close all windows
# # # #     cap.release()
# # # #     cv2.destroyAllWindows()

# # #import libraries 
# # import numpy as np
# # import cv2
# # import streamlit as st 


# # # Step 1: Identify the webcam
# # # For video recording paste the file path of the video

# # webcam = cv2.VideoCapture(0) # Local webcam - 0, External webcam - 1

# # #Load the face cascade classifier
# # #The code loads the face cascade classifier file from the specified path.

# # face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # face_cascade = cv2.CascadeClassifier(face_cascade_name)

# # # eye detection
# # eye_cascade_name = cv2.data.haarcascades + 'haarcascade_eye.xml'
# # eye_cascade = cv2.CascadeClassifier(eye_cascade_name)

# # # Smile detection
# # smile_cascade_name = cv2.data.haarcascades + 'haarcascade_smile.xml'
# # smile_cascade = cv2.CascadeClassifier(smile_cascade_name)

# # #Create a function to capture frames from the webcam and detect faces

# # def detect(gray, frame):
# #     face = face_cascade.detectMultiScale(gray, 1.3, 5)
# #     for (x, y, w, h) in face:
# #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
# #         roi_gray = gray[y:y+h, x:x+w]
# #         roi_color = frame[y:y+h, x:x+w]
        
# #         #Eye Detection
# #         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) #Actual eye detection
# #         for (ex, ey, ew, eh) in eyes:
# #             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 100), 3)
        
# #         # Smile Detection
# #         smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
# #         for (sx, sy, sw, sh) in smile:
# #             cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (100, 55, 160), 3)
        
# #     return frame

# # # Step 2: Switch on the webcam

# # while True:
# #     _, frame = webcam.read() # switch on the webcam 
    
# #     # Convert colored frame to black and white
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# #     video = detect(gray, frame)
    
# #     cv2.imshow('Quantum Analytics Facial Detection', video)
# #     if cv2.waitKey(1) & 0xff == ord('q'):
# #         break

# # webcam.release()
# # cv2.destroyAllWindows()


# # # import numpy as np
# # # import cv2
# # # import streamlit as st 

# # # # Load the face cascade classifier
# # # face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # # face_cascade = cv2.CascadeClassifier(face_cascade_name)

# # # # Load the eye cascade classifier
# # # eye_cascade_name = cv2.data.haarcascades + 'haarcascade_eye.xml'
# # # eye_cascade = cv2.CascadeClassifier(eye_cascade_name)

# # # # Load the smile cascade classifier
# # # smile_cascade_name = cv2.data.haarcascades + 'haarcascade_smile.xml'
# # # smile_cascade = cv2.CascadeClassifier(smile_cascade_name)

# # # # Function to detect faces, eyes, and smiles
# # # def detect_faces(gray, frame):
# # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# # #     for (x, y, w, h) in faces:
# # #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# # #         roi_gray = gray[y:y + h, x:x + w]
# # #         roi_color = frame[y:y + h, x:x + w]

# # #         # Detect eyes within the face region
# # #         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
# # #         for (ex, ey, ew, eh) in eyes:
# # #             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# # #         # Detect smiles within the face region
# # #         smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
# # #         for (sx, sy, sw, sh) in smiles:
# # #             cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

# # #     return frame

# # # # Main function to capture frames from the webcam
# # # def main():
# # #     st.title("Face Detection using Haar Cascades")

# # #     # Initialize the webcam
# # #     webcam = cv2.VideoCapture(0)

# # #     if not webcam.isOpened():
# # #         st.error("Error: Unable to access the webcam.")
# # #         return

# # #     while True:
# # #         ret, frame = webcam.read()
# # #         if not ret:
# # #             st.error("Error: Failed to capture frame from the webcam.")
# # #             break

# # #         # Convert the frame to grayscale
# # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # #         # Detect faces, eyes, and smiles
# # #         detected_frame = detect_faces(gray, frame)

# # #         # Display the detected frame
# # #         st.image(detected_frame, channels="BGR")

# # #         # Check for key press to exit the loop
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break

# # #     # Release the webcam
# # #     webcam.release()

# # # if __name__ == "__main__":
# # #     main()


# import streamlit as st
# import numpy as np
# import cv2

# # Load Haar cascades
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# # Function to detect faces, eyes, and smiles
# def detect_faces(gray, frame):
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         # Eye detection
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
#         # Smile detection
#         smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
#         for (sx, sy, sw, sh) in smiles:
#             cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        
#     return frame


# def main():
#     st.title("Facial Detection with OpenCV and Streamlit")
#     st.write("Press the button below to start detecting faces from your webcam")

#     if st.button("Detect Faces"):
#         webcam = cv2.VideoCapture(0)
#         while webcam.isOpened():
#             ret, frame = webcam.read()
#             if not ret:
#                 st.error("Error: Failed to capture frame from the webcam.")
#                 break
            
#             detected_frame = detect_faces(frame)
#             st.image(detected_frame, channels="BGR", use_column_width=True)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         webcam.release()

# if __name__ == "__main__":
#     main()


import cv2

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect faces, eyes, and smiles
def detect_faces(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        
    return frame
