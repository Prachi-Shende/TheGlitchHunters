from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import winsound
import time

# Load pre-trained gender detection model
model = load_model('gender_detection.model')

# Open webcam
webcam = cv2.VideoCapture(0)

# Class labels
classes = ['man', 'woman']

# Threshold for proximity to consider a woman surrounded by men
proximity_threshold = 250  # Adjust this based on context

# Number of frames to consider for gender locking
gender_lock_threshold = 120  # Example: Lock after 30 consistent frames (~1 second for 30 FPS)

# Store gender predictions and lock status for each face
face_gender_history = {}
locked_gender_map = {}

# Function to raise an alarm
def raise_alarm():
    print("Alarm triggered!")
    winsound.Beep(1000, 1000)  # 1 kHz tone for 1 second on Windows

# Function to assign unique IDs to faces and track them
def assign_face_ids(faces):
    face_ids = []
    for i, face in enumerate(faces):
        face_ids.append(i)  # Simple index assignment, improve with tracking
    return face_ids

# Loop through the frames captured from the webcam
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not access the webcam.")
        break

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Assign IDs to faces
    face_ids = assign_face_ids(faces)

    # Counters and position holders for men and women
    men_positions = []
    women_positions = []

    # Loop through detected faces
    for idx, (face_id, face) in enumerate(zip(face_ids, faces)):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Check if the gender is already locked for this face
        if face_id in locked_gender_map:
            # Use the locked gender
            label = locked_gender_map[face_id]
        else:
            # Crop the detected face
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Ensure the face region is large enough to process
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocess the cropped face for gender detection
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender using the model
            conf = model.predict(face_crop)[0]
            label_idx = np.argmax(conf)
            predicted_gender = classes[label_idx]

            # Store the prediction history
            if face_id not in face_gender_history:
                face_gender_history[face_id] = []

            face_gender_history[face_id].append(predicted_gender)

            # Ensure the history list does not grow too large
            if len(face_gender_history[face_id]) > gender_lock_threshold:
                face_gender_history[face_id] = face_gender_history[face_id][-gender_lock_threshold:]

            # Determine if the gender should be locked
            if len(face_gender_history[face_id]) == gender_lock_threshold:
                # Check if all recent predictions are consistent
                most_common_gender = max(set(face_gender_history[face_id]), key=face_gender_history[face_id].count)
                if face_gender_history[face_id].count(most_common_gender) >= int(0.8 * gender_lock_threshold):
                    # Lock the gender
                    locked_gender_map[face_id] = most_common_gender
                    label = most_common_gender
                else:
                    label = predicted_gender
            else:
                label = predicted_gender

        # Store the positions of detected men and women
        if label == 'man':
            men_positions.append((startX, startY, endX, endY))
        else:
            women_positions.append((startX, startY, endX, endY))

        # Format the label with confidence
        label_text = "{}".format(label)

        # Set position to display the label above the face rectangle
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write the label on the frame
        cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Check if any woman is surrounded by 3 or more men
    for woman in women_positions:
        men_nearby = 0
        wx, wy, _, _ = woman

        for man in men_positions:
            mx, my, _, _ = man

            # Calculate the Euclidean distance between the woman and each man
            distance = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2)

            # Check if the man is within the proximity threshold
            if distance < proximity_threshold:
                men_nearby += 1

        # Trigger the alarm if a woman is surrounded by 3 or more men
        if men_nearby >= 1:
            raise_alarm()
            break

    # Display the video frame with bounding boxes and labels
    cv2.imshow("Gender Detection and Alarm System", frame)

    # Press 'Q' to stop the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
