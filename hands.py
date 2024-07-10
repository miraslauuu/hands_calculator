import cv2 # importing library for computer vision
import mediapipe as mp # importing library with the framework for building pipelines to perform computer vision inference 

#################################################################################################

def count_extended_fingers(hand_landmarks):
    # Extracting the y-coordinates of finger tips and base joints from hand landmarks
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y  # Changed to IP joint
    
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    
    ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    
    pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    # Create a list of finger tip and base y-coordinates
    finger_y_coordinates = [
        ('Thumb', thumb_tip_y, thumb_ip_y),  # Changed to IP joint for thumb
        ('Index', index_tip_y, index_base_y), 
        ('Middle', middle_tip_y, middle_base_y), 
        ('Ring', ring_tip_y, ring_base_y), 
        ('Pinky', pinky_tip_y, pinky_base_y),
        
    ]

    # Count the number of fingers extended (tip y-coordinate higher than base y-coordinate)
    extended_fingers = [name for name, tip_y, base_y in finger_y_coordinates if tip_y < base_y]

    return len(extended_fingers), extended_fingers

#################################################################################################

# Initializing mediapipe hands and OpenCV

mp_hands = mp.solutions.hands # Load MediaPipe Hands module
hands = mp_hands.Hands() # Create a Hands object with default settings
mp_drawing = mp.solutions.drawing_utils # Load drawing utilities

# Opening webcam

cap = cv2.VideoCapture(0) # opening the default camera

operation_result = 0 # Initialize the cumulative sum of extended fingers
previous_num_extended_fingers = 0 # Initialize the previous number of extended fingers

# Continuously capturing the frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) # flipping the frame horizontally for selfie-view display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converting the frame to rgb
    result = hands.process(rgb_frame) # performing hand detection and tracking   

    # Drawing hand landmarks
    if result.multi_hand_landmarks: # if hands are detected
        for hand_landmarks in result.multi_hand_landmarks:
            num_extended_fingers, extended_fingers = count_extended_fingers(hand_landmarks)
            
            # Update the cumulative sum only if the number of extended fingers has changed
            if num_extended_fingers != previous_num_extended_fingers:
                operation_result += num_extended_fingers
                previous_num_extended_fingers = num_extended_fingers

            cv2.putText(frame, f"Fingers: {num_extended_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Extended: {', '.join(extended_fingers)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # then draw landmarks and connections on the original frame

    cv2.putText(frame, f"Sum: {operation_result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame) # show the frame with landmarks

    if cv2.waitKey(1) & 0xFF == ord('q'): # exit on q
        break

cap.release() # release the webcam
cv2.destroyAllWindows() # close OpenCV windows
