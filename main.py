import cv2
import mediapipe as mp
from fer import FER
import time
import collections

# Initialize face detection and emotion recognition
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER(mtcnn=True)

print('hi')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('hi')

# Emotion smoothing variables
emotion_history = collections.deque(maxlen=10)  # Stores last 10 detected emotions
previous_emotion = None  # Tracks last stable emotion

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = frame[y:y+h, x:x+w]

                # Detect emotion
                if face.size != 0:
                    emotion_data = emotion_detector.top_emotion(face)

                    if emotion_data is not None:
                        emotion, score = emotion_data

                        # Ensure score is valid before using it
                        if score is not None and score > 0.5:
                            emotion_history.append(emotion)

                        # Compute the most common emotion in recent frames
                        if emotion_history:
                            most_common_emotion = max(set(emotion_history), key=emotion_history.count)
                        else:
                            most_common_emotion = previous_emotion

                        # Update the displayed emotion only if it's stable
                        if most_common_emotion != previous_emotion:
                            previous_emotion = most_common_emotion

                        # Display the smoothed emotion
                        cv2.putText(frame, f"{previous_emotion}: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.02)  # Small delay to reduce processing load

cap.release()
cv2.destroyAllWindows()
