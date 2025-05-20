import cv2
import json

cap = cv2.VideoCapture(0)

def sl_text():
    import mediapipe as mp
    import numpy as np
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    import time

    # Initialize MediaPipe hands, face detection, and drawing utilities
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Start the camera feed
    model_name = "DCGX-ISL-Alphabet"
    label_encoder = LabelEncoder()
    label_list_path = (model_name+".keras_labels.json")

    with open(label_list_path, "r") as f:
        label_list = json.load(f)
    label_encoder.fit(label_list)

#-------------- Init Variables -------------------------------------------------------------------------
    disp = "['']"
    model = load_model("models/"+model_name+".keras")
    frame_count = 0
    prev_time = time.time()  # time at the start
    fps = 0
    disp_label = ''
    recent_labels = []
    confidence = ""

#---- Use MediaPipe Hands and Face Detection with specific parameters ----------------------------------
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
         mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB as MediaPipe expects RGB input

            # Perform hand and face detection
            hand_results = hands.process(rgb_frame)
            face_results = face_detection.process(rgb_frame)

            # Initialize empty lists for left, right hand points, and head position
            left_list, right_list = [], []
            head_position = None

            # If hands are detected
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Check if it's a left or right hand
                    hand_label = hand_results.multi_handedness[idx].classification[0].label
                    hand_points = []

                    # Collect all normalized coordinates for the current hand
                    for landmark in hand_landmarks.landmark:
                        hand_points.append((landmark.x, landmark.y))

                    # Assign points to the respective list based on hand label
                    if hand_label == 'Left':
                        left_list = hand_points
                    else:
                        right_list = hand_points

                    # Draw hand landmarks and bounding box on the frame
                    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get coordinates for bounding box around the hand
                    x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])

                    # Draw the bounding box around the hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw a black box at the top of the bounding box with "sign" label
                    box_position = (x_min, y_min - 30)
                    cv2.rectangle(frame, box_position, (x_min + 100, y_min), (0, 0, 0), -1)
                    cv2.putText(frame, disp_label[2:-2], (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            # Detect and display face bounding box if a face is detected
            if face_results.detections:
                for detection in face_results.detections:
                    # Extract bounding box information for head position
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * frame.shape[1])
                    y_min = int(bbox.ymin * frame.shape[0])
                    width = int(bbox.width * frame.shape[1])
                    height = int(bbox.height * frame.shape[0])
                    x_max = x_min + width
                    y_max = y_min + height

                    # Approximate head position as the center of the face bounding box
                    head_position = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                    # Draw the bounding box around the head
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    # Draw a black box at the top of the bounding box with "head" label
                    box_position = (x_min, y_min - 30)
                    cv2.rectangle(frame, box_position, (x_min + 50, y_min), (0, 0, 0), -1)
                    cv2.putText(frame, "head", (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if hand_results.multi_hand_landmarks:
                # Create Normalized Positions
                if head_position:
                    normalized_head_x = head_position[0] / frame.shape[1]
                    normalized_head_y = head_position[1] / frame.shape[0]

                    if not left_list:
                        left_list = [(0, 0)] * 21
                    if not right_list:
                        right_list = [(0, 0)] * 21
                    major_list = left_list + right_list + [(normalized_head_x, normalized_head_y)]
                    real_time_input = np.array(major_list).flatten().reshape(1, -1)  # Flatten and reshape for model input

                    # Perform real-time prediction
                    real_time_pred = model.predict(real_time_input)
                    predicted_class = np.argmax(real_time_pred, axis=1)  # Multiclass classification
                    confidence = str(int(np.max(real_time_pred, axis=1)[0]*100))
                    disp_label = str(label_encoder.inverse_transform(predicted_class))
                    print(str(predicted_class), str(label_encoder.inverse_transform(predicted_class)))

                else:
                    print("Head Position: Not Detected")
                    disp_label = "  No_symbol  "

                current_label = disp_label[2:-2]
                recent_labels.append(current_label)

            # Break the loop on 'Esc' key press
            if key == 27:  # ASCII for Esc key is 27
                break

            #Frame Info Draw

            detected_sign_disp = "Sign:" + disp_label[2:-2] + " " + confidence + "%"
            # Draw the Info box
            cv2.rectangle(frame, (0, 0), (300, 100), (50, 0, 0), -1)
            if face_results.detections:
                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    cv2.putText(frame, detected_sign_disp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Sign:None", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No Person", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Increment the frame count
            frame_count += 1

            # Calculate the time difference between frames
            curr_time = time.time()
            time_diff = curr_time - prev_time

            # Calculate FPS if at least one second has passed
            if time_diff >= 1.0:
                fps = frame_count / time_diff
                prev_time = curr_time  # Update time for the next second
                frame_count = 0  # Reset frame count for the next second

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("DCGX Number", frame) # Display the output

sl_text()

# Release resources
cap.release()
cv2.destroyAllWindows()