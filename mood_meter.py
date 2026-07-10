import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def main(argv=None):
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        print("Mood Meter starting. Press 'ESC' to exit.")
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
                
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # To improve performance, mark the image as not writeable
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            image.flags.writeable = True
            
            mood_score = 50 # Default neutral
            mood_text = "Neutral"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Face width (temples: 234, 454) for normalization
                    face_width = calculate_distance(face_landmarks.landmark[234], face_landmarks.landmark[454])
                    
                    # Mouth width (corners: 61, 291)
                    mouth_width = calculate_distance(face_landmarks.landmark[61], face_landmarks.landmark[291])
                    
                    # Eyebrow distance (inner: 107, 336)
                    eyebrow_dist = calculate_distance(face_landmarks.landmark[107], face_landmarks.landmark[336])
                    
                    if face_width > 0:
                        mouth_ratio = mouth_width / face_width
                        eyebrow_ratio = eyebrow_dist / face_width
                        
                        # Basic Calibration
                        if mouth_ratio > 0.40:
                            # Smiling (Happy)
                            mood_score = 50 + ((mouth_ratio - 0.40) / 0.10) * 50
                            mood_text = "Happy"
                        elif eyebrow_ratio < 0.20:
                            # Frowning (Mad/Sad)
                            mood_score = ((eyebrow_ratio - 0.15) / 0.05) * 50
                            mood_text = "Mad / Sad"
                        else:
                            mood_score = 50
                            mood_text = "Neutral"
                        
                        # Clamp between 0 and 100
                        mood_score = max(0, min(100, int(mood_score)))
                        
            # Draw the Mood Meter bar
            bar_width = 300
            bar_height = 30
            x_offset = 50
            y_offset = 80
            
            # Background bar
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (100, 100, 100), -1)
            
            # Fill bar based on score
            fill_width = int((mood_score / 100) * bar_width)
            
            # Color gradient: 0 is red, 50 is yellow, 100 is green
            if mood_score < 50:
                # Red to Yellow
                r = 255
                g = int((mood_score / 50) * 255)
            else:
                # Yellow to Green
                r = int((1 - (mood_score - 50) / 50) * 255)
                g = 255
            b = 0
            
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + fill_width, y_offset + bar_height), (b, g, r), -1)
            
            # Draw border
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (255, 255, 255), 2)
            
            # Add text
            cv2.putText(image, f"Mood: {mood_score}/100", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"State: {mood_text}", (x_offset, y_offset + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Mood Meter', image)
            
            # Exit on ESC or if window is closed
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if cv2.getWindowProperty('Mood Meter', cv2.WND_PROP_VISIBLE) < 1:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
