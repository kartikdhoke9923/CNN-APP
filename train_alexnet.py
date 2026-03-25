import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

print("🚀 Training AlexNet CIFAR-10 (20 mins)...")

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# YOUR EXACT MODEL ARCHITECTURE
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (3,3), input_shape=(32,32,3), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(256, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(384, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Conv2D(384, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Conv2D(256, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# Compile with YOUR settings
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training started (15 epochs)...")
history = model.fit(X_train, y_train, 
                   batch_size=128, 
                   epochs=15, 
                   validation_split=0.2, 
                   verbose=1)

# Test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎉 FINAL TEST ACCURACY: {test_acc*100:.1f}%")

# Save for Flask
model.save('models/alexnet_cifar10_trained.h5')
print("💾 SAVED: models/alexnet_cifar10_trained.h5")
print("✅ READY FOR FLASK APP!")
