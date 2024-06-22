import numpy as np
import albumentations as albu

augmentations = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
    albu.GaussNoise(var_limit=(0.01, 0.5), always_apply=False, p=0.2),
    albu.CoarseDropout(max_holes=10, max_height=12, max_width=12, fill_value=0, always_apply=False, p=0.2),
    albu.Blur(blur_limit=(2, 4), always_apply=False, p=0.2),
])

def augment_images(x_train, y_train):
    x_train_out = []
    y_train_out = []

    for i in range(len(x_train)):
        augmented = augmentations(image=x_train[i], mask=y_train[i])
        x_train_out.append(augmented['image'])
        y_train_out.append(augmented['mask'])

    return np.array(x_train_out), np.array(y_train_out)
