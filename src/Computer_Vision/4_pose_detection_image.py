import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageOps
import h5py

# Print and store the current working directory
working_directory = os.getcwd()
print("Current working directory:", working_directory)

# Patch the model file to remove the problematic "groups" parameter
with h5py.File("src\\ComputerVision\\keras_model.h5", "r+") as f:
    model_config = f.attrs.get("model_config")
    if model_config and '"groups": 1,' in model_config:
        new_config = model_config.replace('"groups": 1,', '')
        f.attrs.modify("model_config", new_config)
        f.flush()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("src\\ComputerVision\\keras_model.h5", compile=False)
class_names = open("src\\ComputerVision\\labels.txt", "r").readlines()

# Clear screen for Windows or Unix-based systems
clear_command = 'cls' if os.name == 'nt' else 'clear'
os.system(clear_command)
print("Current working directory:", working_directory)

# this function will detect negative samples and low confidence poses to filter out images that aren't related to the pose detection model
def analyze_pose(prediction, confidence_threshold=0.5):
    """
    Analyze if the pose is valid
    Returns: (is_valid, message, confidence_score)
    """
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    
    # Only check confidence threshold
    if confidence_score < confidence_threshold:
        return False, "Low confidence - Pose not clear", confidence_score
    
    # All poses are valid if they meet confidence threshold
    return True, "Valid pose detected", confidence_score

while(True):
    # Prompt user for an image path
    image_path = input("Enter the path to an image: ").strip()

    # Check if the file exists
    if not os.path.exists(image_path):
        print("Image file does not exist.")
    else:
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        # Resize and crop the image to 224x224 using LANCZOS resampling
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        # Convert image to numpy array and normalize
        img_array = np.asarray(img, dtype=np.float32)
        normalized_img = (img_array / 127.5) - 1
        # Add batch dimension
        input_data = normalized_img.reshape(1, 224, 224, 3)

        # Perform prediction
        prediction = model.predict(input_data)
        is_valid, message, confidence_score = analyze_pose(prediction)
        
        if is_valid:
            index = np.argmax(prediction)
            predicted_class = class_names[index]
            print("Class:", predicted_class[2:], end=" ")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        else:
            print(message)
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            print("Please try a different pose")