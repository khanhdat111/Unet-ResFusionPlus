from tensorflow.keras.callbacks import EarlyStopping , ReduceLROnPlateau, ModelCheckpoint
import gc
from CustomLayers.DataAugmentaion import augment_images
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture.Unet_ResFusionPlus import create_model
from ImageLoader.ImageLoader2D import load_data

#Prepare data:
train_path = "path_your_train_data"
val_path = "path_your_valid_data"
test_path = "path_your_test_data"

img_height, img_width = 352, 352

X_train, Y_train = load_data(img_height, img_width, "train", train_path)
X_val, Y_val = load_data(img_height, img_width, "validation", val_path)
X_test, Y_test = load_data(img_height, img_width, "test", test_path)

#Training
learning_rate = 1e-4

optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay= 1e-4)
model.compile(optimizer=optimizer, loss=dice_metric_loss , metrics=['acc'])

EPOCHS = 1000
min_loss_for_saving = 0.2

for epoch in range(EPOCHS):

    print(f'Training, epoch {epoch+1}')
    print(f'Learning Rate: {learning_rate}')
    
    X_train_augmented, Y_train_augmented = augment_images(X_train, Y_train)

    model.fit(x=X_train_augmented, y=Y_train_augmented, epochs=1, batch_size=4,
              validation_data=(X_val, Y_val), verbose=1)
        
    prediction_valid = model.predict(X_val, verbose=0)
    loss_valid = dice_metric_loss(Y_val, prediction_valid).numpy()

    print("Loss Validation: " + str(loss_valid))

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved model with val_loss: ", loss_valid)
        model.save_weights(f"/kaggle/working/best_model.weights.h5")
        
    del X_train_augmented
    del Y_train_augmented
        
    gc.collect()
