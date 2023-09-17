import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image
img = Image.open("image.png").convert('RGB')

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Build a Convolutional Neural Network
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("my_cifar10_model.keras")

# Load the model from file
loaded_model = tf.keras.models.load_model("my_cifar10_model.keras")

# Load and preprocess the image
img = Image.open("image.png")  # Load the image
img = img.resize((32, 32))  # Resize to 32x32
img_array = np.array(img)  # Convert to a NumPy array
img_array = img_array / 255.0  # Normalize
img_array = img_array.reshape(1, 32, 32, 3)  # Reshape for model input

# Use the model to make a prediction
predictions = loaded_model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

# Map the predicted class index to its corresponding name
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class_name = class_names[predicted_class_index]

# Output the result
print(f"Predicted class: {predicted_class_name}")
