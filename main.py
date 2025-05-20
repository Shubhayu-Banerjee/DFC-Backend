from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the DeafCeption Demo!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model("DCGX-ISL-Alphabet.keras")

# Initialize label encoder
label_encoder = LabelEncoder()

with open("DCGX-ISL-Alphabet.keras_labels.json", "r") as f:
    label_list = json.load(f)
label_encoder.fit(label_list)

label_encoder.fit(['Energy (Left)', 'Energy (Right)', 'Good (Left)', 'Good (Right)', 'Hi (Left)', 'Hi (Right)', 'Me (Left)', 'Me (Right)', 'Please (Left)', 'Please (Right)', 'Thank You (Left)', 'Thank You (Right)']
                  )

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = mp_hands.process(rgb_frame)
            face_results = mp_face.process(rgb_frame)

            left_list, right_list = [], []
            head_position = None

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = hand_results.multi_handedness[idx].classification[0].label
                    hand_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    if hand_label == 'Left':
                        left_list = hand_points
                    else:
                        right_list = hand_points

            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * frame.shape[1])
                    y_min = int(bbox.ymin * frame.shape[0])
                    width = int(bbox.width * frame.shape[1])
                    height = int(bbox.height * frame.shape[0])
                    x_max = x_min + width
                    y_max = y_min + height
                    head_position = ((x_min + x_max) // 2, (y_min + y_max) // 2)

            disp_label = ""
            if hand_results.multi_hand_landmarks and head_position:
                norm_head_x = head_position[0] / frame.shape[1]
                norm_head_y = head_position[1] / frame.shape[0]

                if not left_list:
                    left_list = [(0, 0)] * 21
                if not right_list:
                    right_list = [(0, 0)] * 21
                major_list = left_list + right_list + [(norm_head_x, norm_head_y)]
                model_input = np.array(major_list).flatten().reshape(1, -1)

                prediction = model.predict(model_input)
                pred_class = np.argmax(prediction, axis=1)
                disp_label = label_encoder.inverse_transform(pred_class)[0]

            await websocket.send_text(disp_label or "No_symbol")
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")