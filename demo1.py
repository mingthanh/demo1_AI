import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# tai, chuan bi MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# chuan hoa du lieu (normalize)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# bien dich
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# rate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"do chinh xac: {test_acc}")

# du doan KQ
predictions = model.predict(test_images)

# hien thi HA va du doan
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"du doan: {predictions[i].argmax()}")
    plt.show()
