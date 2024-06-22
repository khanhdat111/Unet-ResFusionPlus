import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.io import imread

def load_data(img_height, img_width, data_type, dataset_path):
    IMAGES_PATH = dataset_path + 'images/'
    MASKS_PATH = dataset_path + 'masks/'

    image_files = glob.glob(IMAGES_PATH + "*.jpg")
    images_to_be_loaded = len(image_files)

    print(f'Resizing and augmenting {data_type} images and masks: {images_to_be_loaded}')

    X = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    for n, image_path in tqdm(enumerate(image_files), total=images_to_be_loaded):
        mask_path = image_path.replace('images', 'masks')

        image = imread(image_path)
        mask = imread(mask_path)
        
        pillow_image = Image.fromarray(image)
        image_resized = np.array(pillow_image.resize((img_width, img_height))) / 255.0

        pillow_mask = Image.fromarray(mask)
        mask_resized = np.array(pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS))

        binary_mask = (mask_resized >= 127).astype(np.uint8)

        X[n] = image_resized
        Y[n] = binary_mask

    Y = np.expand_dims(Y, axis=-1)

    return X, Y

# train_path = "your_path_train"
# val_path = "your_path_validation"
# test_path = "your_path_test"
