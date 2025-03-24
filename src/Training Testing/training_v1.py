import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import datetime

# -------------------------------
# Parameters & File Paths
# -------------------------------
img_height, img_width = 224, 224      # Dimensions expected by MobileNetV2
batch_size = 32
epochs = 10                         # Adjust number of epochs as needed

# Your training data folder (ensure it contains subfolders for each pose/category)
data_dir = r"C:\Users\Colin\OneDrive\Pictures\bb_training_data"

# -------------------------------
# Data Preparation with Augmentation
# -------------------------------
# Use validation_split to automatically reserve a portion of data for validation.
datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values to [0,1]
    rotation_range=20,             # Random rotations for augmentation
    width_shift_range=0.2,         # Random horizontal shifts
    height_shift_range=0.2,        # Random vertical shifts
    shear_range=0.15,              # Shear transformation
    zoom_range=0.15,               # Random zoom
    horizontal_flip=True,          # Random horizontal flip
    fill_mode="nearest",           # How to fill newly created pixels
    validation_split=0.2           # Reserve 20% of images for validation
)

# Create training generator (using 80% of the data)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',    # Assumes multiple categories (poses)
    subset='training',           # Use this subset for training
    shuffle=True
)

# Create validation generator (using the remaining 20% of the data)
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',         # Use this subset for validation
    shuffle=True
)

# -------------------------------
# Model Definition Using Transfer Learning
# -------------------------------
# Load MobileNetV2 with pre-trained ImageNet weights and exclude the top layers.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model's layers so they are not updated during training.
base_model.trainable = False

# Add new layers on top of the base model.
x = base_model.output
x = GlobalAveragePooling2D()(x)         # Reduce spatial dimensions to a vector
x = Dense(128, activation='relu')(x)      # Fully-connected layer with 128 neurons
x = Dropout(0.5)(x)                       # Dropout for regularization to avoid overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Final classification layer

# Combine the base model and the new layers into one model.
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer and categorical crossentropy loss.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary to see the architecture.
model.summary()

# -------------------------------
# Training the Model
# -------------------------------
# Fit the model using the training and validation generators.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# -------------------------------
# Save the Trained Model to a Relative Path
# -------------------------------
# Get the directory where the script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative directory for saving the model ("Custom Models" folder within the current folder)
save_dir = os.path.join(script_dir, "Custom Models")
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist

# Get today's date to include in the filename (format: YYYY-MM-DD)
today = datetime.datetime.today().strftime("%Y-%m-%d")
model_filename = f"bb_pose_model_{today}.h5"

# Define the full path for saving the model
save_path = os.path.join(save_dir, model_filename)
model.save(save_path)
print(f"Model saved to {save_path}")
