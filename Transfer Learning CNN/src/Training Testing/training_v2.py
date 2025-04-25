import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import datetime

# -------------------------------
# Parameters & File Paths
# -------------------------------
print("Setting parameters and file paths...")
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

data_dir = r"C:\Users\Colin\OneDrive\Pictures\bb_training_data"

# -------------------------------
# Data Preparation with Augmentation
# -------------------------------
print("Starting data preparation with augmentation...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

print("Creating training generator...")
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Creating validation generator...")
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# -------------------------------
# Manual Class Weights (Aggressively Reduce Not-a-Pose Bias)
# -------------------------------
print("Applying manual class weights (reduce 'not a pose' bias)...")
# Assuming class 0 is "not a pose", and other class indices are 1–4
class_weights = {
    0: 0.1,  # Not a Pose (cranked down)
    1: 1.0,  # Front Double Bi
    2: 1.0,  # Back Double Bi
    3: 1.0,  # Side Chest
    4: 1.0   # Abs and Thighs
}
print("Manual class weights used:", class_weights)

# -------------------------------
# Model Definition Using Transfer Learning
# -------------------------------
print("Loading base model (MobileNetV2) with pre-trained ImageNet weights...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

print("Adding custom layers...")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model summary:")
model.summary()

# -------------------------------
# Training the Model
# -------------------------------
print("Starting model training...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    class_weight=class_weights  # << Manual weights applied
)
print("Training complete.")

# -------------------------------
# Save the Trained Model
# -------------------------------
print("Preparing to save the model...")
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "Custom Models")
os.makedirs(save_dir, exist_ok=True)

today = datetime.datetime.today().strftime("%Y-%m-%d")
model_filename = f"bb_pose_model_{today}.h5"
save_path = os.path.join(save_dir, model_filename)
model.save(save_path)
print(f"Model saved to {save_path}")

# -------------------------------
# Generate and Save labels.txt
# -------------------------------
print("Generating labels file...")
class_indices = train_generator.class_indices
sorted_classes = sorted(class_indices.items(), key=lambda x: x[1])
labels_filename = f"labels_{today}.txt"
labels_path = os.path.join(save_dir, labels_filename)

with open(labels_path, "w") as f:
    for class_name, idx in sorted_classes:
        f.write(f"{idx} {class_name}\n")
print(f"Labels saved to {labels_path}")
