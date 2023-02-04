import os
import numpy as np
import tensorflow as tf
from PIL import Image
import ratings

def run_train_test(ratings, training_input, testing_input):
    # MODEL
    from keras.api._v2.keras import Sequential, Conv2D, MaxPooling2D, Flatten, Dense

    model = Sequential()

    # number of filters, size of filters, stride
    model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256,256,3)))
    # max value after the relu activation (condensing the number)
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    # condensing it to a single value
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss = tf.losses.BinaryConssentropy(), metrics=['accuracy'])

    model.summary()
    # TRAINING DATA

    # TESTING DATA
    """
    for cat in testing_input:
        print("The rating that was given to " + cat + " was " + classify(testing_input[cat]))
        print("The actual result was " + ratings[cat])
    """
    
    pass

def image_array(path):
    folder = os.listdir(path)
    images = {}

    for filename in folder:
        file = os.path.join(path, filename)
        image = Image.open(file)

        img_resize = image.resize((256, 256))

        img_arr = np.asarray(img_resize)
        images[filename] = img_arr

    return images


if __name__ == "__main__":
    import sys
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    train_data = image_array(sys.argv[1])
    test_data = image_array(sys.argv[2])

    #print(len(train_data))
    #print(len(test_data))
    
    run_train_test(ratings.cat_rating, train_data, test_data)