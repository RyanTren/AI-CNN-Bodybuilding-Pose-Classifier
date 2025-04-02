# CS3642 AI Semester Project - AI-Image Processing Model on Predicting Winning Bodybuilder Contestants

## directory structure

AI-Based-Fake-Job-Posting-Detector/
└── src/
    └── ComputerVision/
        ├── good_image/
        │   ├── pose1.jpg        # Good bodybuilding poses
        │   ├── pose2.jpg
        │   └── ...
        ├── bad_image/
        │   ├── bad1.jpg         # Bad/incorrect bodybuilding poses
        │   ├── bad2.jpg
        │   └── ...
        ├── keras_model.h5
        ├── labels.txt
        └── 4_pose_detection_image.py


- good_image/ folder should contain:
    * Correctly executed bodybuilding poses
    * Clear, well-lit photos
    * Proper form and stance
    * Professional competition poses

- bad_image/ folder should contain:
    * Incorrect pose executions
    * Poor form examples
    * Blurry or poorly lit photos
    * Common mistakes in poses


## requirements to run model!
- First have conda installed, then create an environment in the project's terminal.

```ps
conda init powershell
conda --version
```

```ps
conda create -n new-cv-env python=3.10
conda activate new-cv-env
conda install -c conda-forge opencv
conda install tensorflow
conda install pillow h5py
```

- This will help us install all the tensorflow dependencies used to run the Keras Computer Vision model

Some takeway on our first implementation: 
- accuracy is not there... 
- need to use a better model or grow our data set


## Improvement Log:
### Improve Data Quality (Fastest Impact):
* Focus on quality over quantity
* Use high-quality competition photos
* Ensure consistent lighting and angles

```python
import tensorflow as tf

def create_augmented_dataset(directory, batch_size=32):
    """
    Create a dataset with augmentation specifically tuned for bodybuilding poses
    """
    # Create the base dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="training"
    )

    # Create validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="validation"
    )

    # Enhanced augmentation for bodybuilding poses
    def augment_images(x, y):
        # Random flip - good for pose symmetry
        x = tf.image.random_flip_left_right(x)
        
        # Slight rotation - accounts for camera angle variations
        x = tf.image.random_rotation(x, 0.1)  # 10% rotation max
        
        # Brightness/contrast - handles different lighting conditions
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        
        # Slight zoom - handles different distances
        x = tf.image.random_zoom(x, 0.1)
        
        return x, y

    # Apply augmentation and optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (
        dataset
        .map(augment_images, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, val_dataset

# Usage example:
def train_model():
    # Create datasets
    train_ds, val_ds = create_augmented_dataset(
        "path/to/pose/images",
        batch_size=32
    )

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds
```

### Add data augmentation:

### Use Transfer Learning (Quick Win):

### Improve Pose Detection Logic:

### Add Basic Feature Extraction:

### Implement Simple Ensemble Method:

### Priority Order:
* Use transfer learning with MobileNetV2 or EfficientNetB0 (quick and effective)
* Implement basic data augmentation
* Add pose validation checks
* Improve the confidence threshold system
* Add basic feature extraction if time permits


