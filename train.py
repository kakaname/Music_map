import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def load_images_from_directory(directory_path, target_size=(128, 128)):
    image_files = glob.glob(os.path.join(directory_path, '**', '*.png'), recursive=True)
    images = []
    
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')  # Convert to RGB
        #image = image.resize() Could be used later for better data
        image_array = np.array(image)  # Convert to numpy array
        images.append(image_array)
    
    images = np.array(images)
    return images

# Set the directory path and target size
directory_path = './Data/images_original'
target_size = (432,288)

# Load and preprocess the images
images = load_images_from_directory(directory_path, target_size)
images = images / 255.0  # Normalize pixel values to [0, 1]



# Split the data into training, validation, and test sets
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)


# Now train_images, val_images, and test_images are ready for training
print(f'Training images shape: {train_images.shape}')
print(f'Validation images shape: {val_images.shape}')
print(f'Test images shape: {test_images.shape}')

latent_dim = 64
autoencoder = Autoencoder(latent_dim, (288,432,3))

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(train_images, train_images,
                epochs=1,
                shuffle=True,
                validation_data=(test_images, test_images))


encoded_imgs = autoencoder.encoder(test_images).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(test_images[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

