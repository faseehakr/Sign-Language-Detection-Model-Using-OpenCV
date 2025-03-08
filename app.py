from function import *
from keras.utils import to_categorical 
from keras.models import model_from_json 
from keras.layers import LSTM, Dense 
from keras.callbacks import TensorBoard 
import cv2

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = [(245, 117, 16) for _ in range(20)]
print(len(colors))

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.7) as hands:  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)
        
        print(results)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        
        print(f"Keypoints: {keypoints}")

        if keypoints is not None:
            sequence.append(keypoints)
            sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if len(predictions) > 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

                frame = prob_viz(res, actions, frame, colors, threshold)
        except Exception as e:
            print(f"Error during prediction: {e}")
            pass

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ' ' + ' '.join(accuracy), (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
