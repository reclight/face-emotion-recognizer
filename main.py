import cv2
import mediapipe as mp
from fer import FER

# Initialize face detection and emotion recognition
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER(mtcnn=True)

print('hi')

cap = cv2.VideoCapture(0)
print('hi')

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
                    emotion, score = emotion_detector.top_emotion(face)
                    cv2.putText(frame, f"{emotion}: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Face Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
