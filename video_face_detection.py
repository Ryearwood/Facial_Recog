#!/usr/bin/env python
# coding: utf-8

# !pip install --user -r requirements.txt

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


# In[3]:
# # Test initial webcam connection

# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)

# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False

# while rval:
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#         break

# cv2.destroyWindow("preview")
# vc.release()

# In[4]:
# ## test Facial Detection

# # Get a reference to webcam 
# video_capture = cv2.VideoCapture(0)

# # Initialize variables
# face_locations = []

# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()

#     # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces in the current frame of video
#     face_locations = face_recognition.face_locations(rgb_frame)

#     # Display the results
#     for top, right, bottom, left in face_locations:
#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#     # Display the resulting image
#     cv2.imshow('Video', frame)

#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
# In[5]:
# Modify Code for Video Stream image capture

KNOWN_FACES_DIR = 'dataset/Known_faces'
TOLERANCE = 0.55 # Lower == More Strict
FRAME_THICKNESS = 3
FONT_THICKNESS = 1
MODEL = 'hog'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

video = cv2.VideoCapture("videos/recordings/Original_video_for_demo.mp4") # put filename here also

known_faces = []
known_names = []

# In[6]:
# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [((ord(c.lower())-97)*8) +10 for c in name[:3]]
    return color


# In[7]:
# Evaluate and Encode Known Faces as base for comparison
print('Loading known faces...')
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # face_recognition always returns a list of found faces, for this purpose we take first face only
        # (assuming one face per image in Known_faces as a single person can't appear twice in one image)
        encoding = face_recognition.face_encodings(image)[0]
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
# In[8]:
# Execution loop for fetching and processing video data
print('Processing unknown faces in video feed...')
while True:
 
    ret, image = video.read() # Grab video frames
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert it from RGB to BGR as we are going to work with cv2
    locations = face_recognition.face_locations(image, model=MODEL) #grab face locations - we'll need them to draw boxes
    encodings = face_recognition.face_encodings(image, locations) # Pass face locations to be encoded so that model does not need to search again
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert it from BGR to RGB to return original image colouring
    # We assume that there might be more faces in an image - let's find the faces of different people
    for face_encoding, face_location in zip(encodings, locations): 

        # We use compare_faces (but can use face_distance for distance-score measurement as well)
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE) # Returns array of True/False values in order of passed known_faces
        
        # Order is preserved, so we check if any face was found then grab its index, then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            # print(f' - {match} from {results}')
            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            # Get color by name using our fancy function
            color = name_to_color(match)
            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            # Create a smaller, filled frame below for a name
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22) # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), FONT_THICKNESS)

    # Display video in real-time
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release() # Release Webcam Handle
cv2.destroyAllWindows() # close all windows upon code termination