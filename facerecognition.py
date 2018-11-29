import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import pickle

# adapted from the Deep Learning: Face Recognition course from LinkedIn Learning

IMAGES_PATH = './intranet_images'  # put your reference images in here
#IMAGES_PATH = './images'    # put your reference images in here
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/brim.avi', fourcc, 7.0, (640, 480))

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def load_database_from_file():
    with open('database', 'rb') as fp:
        database = pickle.load(fp)

    return database

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}
    count = 0

    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]

        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)

        if len(encodings) == 0:
            print(f'Face encodings not found for user {identity}.')
        else:
            print(f'Encoding face for user #{count}: {identity}')
            database[identity] = encodings[0]
            count = count + 1

    with open('database', 'wb') as fp:
        pickle.dump(database, fp)
    return database

def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


def run_face_recognition(database):
    """
    Start the face recognition via the webcam
    """
    # Open a handler for the camera
    video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)
    #video_capture.set(3, 900)
    #video_capture.set(4, 1600)
    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    while video_capture.isOpened():
        # Grab a single frame of video (and check if it went ok)
        ok, frame = video_capture.read()
        if not ok:
            logging.error("Could not read frame from camera. Stopping video capture.")
            break

        # flip image
        frame = cv2.flip(frame, 1)

        # run detection and embedding models
        face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

        # Loop through each face in this frame of video and see if there's a match
        for location, face_encoding in zip(face_locations, face_encodings):

            # get the distances from this encoding to those of all reference images
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # select the closest match (smallest distance) if it's below the threshold value
            if np.any(distances <= MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
            else:
                name = None

            # put recognition info on the image
            paint_detected_face_on_image(frame, location, name)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # write the flipped frame
        out.write(frame)


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


database = setup_database()
# if you want to load the face encoding from a file instead of processing it at each run, use the function load_database_from_file()
#database = load_database_from_file()
run_face_recognition(database)