import os
import numpy as np
from PIL import Image
import ratings


def run_train_test(ratings, training_input, testing_input):
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

    train_data = image_array(sys.argv[1])
    test_data = image_array(sys.argv[2])

    print(len(train_data))
    print(len(test_data))

    run_train_test(ratings.cat_rating, train_data, test_data)