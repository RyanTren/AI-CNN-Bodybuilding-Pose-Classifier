from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os
print("Current working directory:", os.getcwd())

import h5py

# with h5py.File("src\\ComputerVision\\keras_model.h5", "r+") as f:
#     model_config = f.attrs.get("model_config")
#     # Since model_config is a string, use a string literal for the search
#     if model_config and '"groups": 1,' in model_config:
#         new_config = model_config.replace('"groups": 1,', '')
#         f.attrs.modify("model_config", new_config)
#         f.flush()


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("src\\Training Testing\\Custom Models\\bb_pose_model_2025-04-02.h5", compile=False)
class_names = open("src\\Training Testing\\Custom Models\\labels_2025-04-02.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Clear screen for Windows or Unix-based systems
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    os.system(clear_command)

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
