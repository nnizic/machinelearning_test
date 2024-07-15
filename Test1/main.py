"""  K  Neuralna mreža  """


import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#    tf.config.experimental.set_memory_growth(device, True)
# postavljanje dataseta slika
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    "flower_photos.tar", origin=dataset_url, extract=True
)
data_dir = pathlib.Path(data_dir).with_suffix("")

# prebrojavanje slika
image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)

# stvaranje dataseta
batch_size = 16
img_height = 64
img_width = 64

# razdvajanje dataseta na trening(80%) i test(20%)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# provjera imena klasa
class_names = train_ds.class_names
print(class_names)

# potrebno koristiti da se I/O ne bi bloako
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardiziranje vrijednosti na [0,1] sa [0,255] (rgb) raspona
normalization_layer = layers.Rescaling(1.0 / 255)

# korištenje ovo layera
# provjera jesu li vrijednosti pixela [0,1]
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))


# osnovni keras model
# za smanjivanje overfittinga dodan je dropout
# ( stavlja aktivaciju na 0 određenom broju trening slika(20%) )

num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="sigmoid"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="sigmoid"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="sigmoid"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="sigmoid"),
        layers.Dense(num_classes),
    ]
)


# kommpajliranje modela
# optimizirannje -> adam
# loss funkcija -> sparsecategoricalcrossentropy

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# pregled scih layera mreže

model.summary()

# treniranje modela

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

# vizualizacija gubitka i točnosti na tring i test setovima

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Točnost treinga")
plt.plot(epochs_range, val_acc, label="Točnost testova")
plt.legend(loc="lower right")
plt.title("Točnost treninga i testova")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Gubitak u treningu")
plt.plot(epochs_range, val_loss, label="Gubitak u  testovima")
plt.legend(loc="upper right")
plt.title("Gubitak treninga i testova")
plt.show()

# predviđanje prema novim podatcima

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)

img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # stvori batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
    f" Ova slika je navjerovatnije iz klase: {class_names[np.argmax(score)]}, sa vjerojatnošću od {100*np.max(score)}%."
)
