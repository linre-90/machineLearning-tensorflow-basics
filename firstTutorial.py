import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# pre prepared dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# classes 0->9
class_names = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# scaling pixel values to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()
"""

allreadyTrained = True

if not allreadyTrained:
    # Construct model from layers
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
        ])

    # Compile model to add optimizer and loss
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    # Summary of the model.
    print(model.summary())

    # Save models weigths in checkpoint. These can be loaded and training be continued.
    checkpoint_path = "cp_weigths/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # callback for saving weigths only
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Train the model
    model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback])

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("\nTest accuracy: ", test_acc)

    # Save the model.
    model.save("firstTutorialModel.h5")

if allreadyTrained:
    # Use saved model
    new_model = tf.keras.models.load_model("firstTutorialModel.h5")
    loaded_test_loss, loaded_test_acc = new_model.evaluate(test_images, test_labels, verbose=2)

    print("Restored model accuracy: ", loaded_test_acc * 100)
    print("Restored model loss: ", loaded_test_loss * 100)

    # Play with model
    indexOfItem = 0 

    # index from commandline
    if len(sys.argv) > 1 :
        indexOfItem = int(sys.argv[1])
       
    propability_model = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])
    predictions = propability_model.predict(test_images)
    plt.imshow(test_images[indexOfItem], cmap="binary")
    plt.title("First tutorial ai thinks it is a: " + class_names[np.argmax(predictions[indexOfItem])])
    plt.show()


















