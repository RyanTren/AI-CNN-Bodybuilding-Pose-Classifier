import tensorflow as tf  # Import the TensorFlow library
print("TensorFlow version:", tf.__version__)  # Print the TensorFlow version to verify installation

# Load the MNIST dataset of handwritten digits
mnist = tf.keras.datasets.mnist

# Split the dataset into training and testing sets.
# x_train and x_test contain the image data, while y_train and y_test contain the corresponding labels.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the image data to a [0, 1] range by dividing by 255.0.
# This helps the neural network to train more efficiently.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a Sequential model, which is a linear stack of layers.
model = tf.keras.models.Sequential([
  # Flatten layer converts each 28x28 image into a 1D array of 784 pixels.
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # Dense layer with 128 neurons and ReLU activation function introduces non-linearity.
  tf.keras.layers.Dense(128, activation='relu'),
  # Dropout layer randomly sets 20% of the inputs to 0 during training to prevent overfitting.
  tf.keras.layers.Dropout(0.2),
  # Final Dense layer with 10 neurons (one per class). No activation is applied here because
  # we will use the loss function that expects raw logits.
  tf.keras.layers.Dense(10)
])

# Run the model on the first training image to get the raw output (logits).
predictions = model(x_train[:1]).numpy()
print("Raw model predictions (logits) for the first image:", predictions)

# Apply the softmax function to convert logits into probabilities.
# This shows the probability distribution over the 10 classes.
softmax_predictions = tf.nn.softmax(predictions).numpy()
print("Softmax probabilities for the first image:", softmax_predictions)

# Define the loss function.
# SparseCategoricalCrossentropy is used for multi-class classification.
# 'from_logits=True' indicates that the model outputs raw logits.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_fn(y_train[:1], predictions).numpy()
print("Initial loss for the first training sample:", initial_loss)

# Compile the model by specifying the optimizer, loss function, and metrics to monitor.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model on the training dataset for 5 epochs.
model.fit(x_train, y_train, epochs=5)

# Evaluate the trained model on the test dataset and print the results.
model.evaluate(x_test,  y_test, verbose=2)

# Create a new Sequential model that adds a Softmax layer on top of the trained model.
# This new model converts logits to probabilities, which can be easier to interpret.
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Use the probability model to predict the class probabilities for the first 5 test images.
print("Predicted probabilities for the first 5 test images:")
print(probability_model(x_test[:5]))
