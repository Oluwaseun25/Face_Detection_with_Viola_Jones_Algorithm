# # # # import streamlit as st
# # # # from Face import detect

# # # # def app():
# # # #     st.title("Face Detection using Viola-Jones Algorithm")
# # # #     st.write("Press the button below to start detecting faces from your webcam")
    
# # # #       #Add a button to start detecting faces
# # # #     if st.button("Detect Faces"):
# # # #         # Call the detect_faces function
# # # #         detect()

# # # # if __name__ == "__main__":
# # # #     app()

# # # import streamlit as st
# # # import cv2
# # # from Face import detect

# # # def app():
# # #     st.title("Face Detection using Viola-Jones Algorithm")
# # #     st.write("Press the button below to start detecting faces from your webcam")
    
# # #     #Add a button to start detecting faces
# # #     if st.button("Detect Faces"):
# # #         # Initialize the webcam
# # #         webcam = cv2.VideoCapture(0)  # Local webcam - 0, External webcam - 1
        
# # #         # Step 2: Switch on the webcam
# # #         while True:
# # #             _, frame = webcam.read()  # switch on the webcam 
            
# # #             # Convert colored frame to black and white
# # #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
# # #             # Call the detect function
# # #             detected_frame = detect(gray, frame)
            
# # #             # Display the detected frame
# # #             st.image(detected_frame, channels="BGR")
            
# # #             if cv2.waitKey(1) & 0xff == ord('q'):
# # #                 break
        
# # #         # Release the webcam
# # #         webcam.release()

# # # if __name__ == "__main__":
# # #     app()

# # import streamlit as st
# # import cv2
# # from Face import detect_faces  # Import detect_faces function from Face.py

# # def app():
# #     st.title("Face Detection using Viola-Jones Algorithm")
# #     st.write("Press the button below to start detecting faces from your webcam")
    
# #     # Add a button to start detecting faces
# #     if st.button("Detect Faces"):
# #         # Initialize the webcam
# #         webcam = cv2.VideoCapture(0)  # Local webcam - 0, External webcam - 1
        
# #         # Loop to capture frames and detect faces
# #         while True:
# #             # Capture frame from webcam
# #             ret, frame = webcam.read()
# #             if not ret:
# #                 st.error("Error: Failed to capture frame from the webcam.")
# #                 break
            
# #             # Convert colored frame to grayscale
# #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
# #             # Detect faces in the frame
# #             detected_frame = detect_faces(gray, frame)  # Call detect_faces function
            
# #             # Display the detected frame
# #             st.image(detected_frame, channels="BGR")
            
# #             # Check if user has stopped the detection
# #             if not st.session_state.detect_faces:
# #                 break
        
# #         # Release the webcam
# #         webcam.release()

# # if __name__ == "__main__":
# #     app()

# import streamlit as st
# import cv2
# from Face import detect_faces  # Import detect_faces function from Face.py

# def app():
#     st.title("Face Detection using Viola-Jones Algorithm")
#     st.write("Press the button below to start detecting faces from your webcam")
    
#     # Initialize session state if it doesn't exist
#     if 'detect_faces' not in st.session_state:
#         st.session_state.detect_faces = False
    
#     # Add a button to start detecting faces
#     if st.button("Detect Faces"):
#         st.session_state.detect_faces = True  # Set detect_faces to True
        
#         # Initialize the webcam
#         webcam = cv2.VideoCapture(0)  # Local webcam - 0, External webcam - 1
        
#         # Loop to capture frames and detect faces
#         while True:
#             # Capture frame from webcam
#             ret, frame = webcam.read()
#             if not ret:
#                 st.error("Error: Failed to capture frame from the webcam.")
#                 break
            
#             # Convert colored frame to grayscale
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces in the frame
#             detected_frame = detect_faces(gray, frame)  # Call detect_faces function
            
#             # Display the detected frame
#             st.image(detected_frame, channels="BGR")
            
#             # Check if user has stopped the detection
#             if not st.session_state.detect_faces:
#                 break
        
#         # Release the webcam
#         webcam.release()

# if __name__ == "__main__":
#     app()

import streamlit as st
import cv2
from Face import detect_faces

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    
    # Initialize session state if it doesn't exist
    if 'detect_faces' not in st.session_state:
        st.session_state.detect_faces = False
    
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        st.session_state.detect_faces = True  # Set detect_faces to True
        
        # Initialize the webcam
        webcam = cv2.VideoCapture(0)  # Local webcam - 0, External webcam - 1
        
        # Loop to capture frames and detect faces
        while True:
            # Capture frame from webcam
            ret, frame = webcam.read()
            if not ret:
                st.error("Error: Failed to capture frame from the webcam.")
                break
            
            # Convert colored frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            detected_frame = detect_faces(gray, frame)  # Call detect_faces function
            
            # Display the detected frame
            st.image(detected_frame, channels="BGR")
            
            # Check if user has stopped the detection
            if not st.session_state.detect_faces:
                break
        
        # Release the webcam
        webcam.release()

if __name__ == "__main__":
    app()

