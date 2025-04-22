import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# Load the model and class labels
# -------------------------------
model_path = "src\\Training Testing\\Custom Models\\(OFFICIAL)bb_pose_model_2025-04-20.h5"
label_path = "src\\Training Testing\\Custom Models\\labels_2025-04-02.txt"

print("Loading model...")
model = load_model(model_path, compile=False)

print("Loading class names...")
class_names = [line.strip().split(' ', 1)[1] for line in open(label_path, "r").readlines()]

# -------------------------------
# Set up validation data generator
# -------------------------------
img_height, img_width = 224, 224
batch_size = 32
validation_dir = r"C:\Users\Colin\OneDrive\Pictures\bb_training_data"

print("Preparing validation generator...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -------------------------------
# Evaluate and visualize results
# -------------------------------
def evaluate_model(model, validation_generator, class_names, save_dir="./evaluation"):
    os.makedirs(save_dir, exist_ok=True)

    print("Generating predictions...")
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    print(report)

    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Evaluation complete. Confusion matrix saved to: {os.path.abspath(save_dir)}")

# Run the evaluation
evaluate_model(model, validation_generator, class_names)
