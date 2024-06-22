from sklearn.metrics import jaccard_score, accuracy_score, f1_score
from tensorflow.keras.optimizers import AdamW

from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture.Unet_ResFusionPlus import create_model
from ImageLoader.ImageLoader2D import load_data

#Prepare data:
test_path = "path_your_test_data"

img_height, img_width = 352, 352
starting_filters = 34
input_chanels = 3
out_classes = 1

X_test, Y_test = load_data(img_height, img_width, "test", test_path)

#Load model:
learning_rate = 1e-4
optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay= 1e-4)
save_path = "path_your_weights"
model = create_model(img_height=img_height, img_width=img_width, input_chanels=input_chanels, out_classes=out_classes, starting_filters=starting_filters)
model.load_weights(save_path)
model.compile(optimizer=optimizer, loss=dice_metric_loss, metrics=["accuracy"])

#Prediction
print("Loading the model")

prediction_test = model.predict(X_test, batch_size=4)

print("Predictions done")

dice_test = f1_score(np.ndarray.flatten(np.array(Y_test, dtype=bool)),
                          np.ndarray.flatten(prediction_test > 0.5))

print("Dice finished")

miou_test = jaccard_score(np.ndarray.flatten(np.array(Y_test, dtype=bool)),
                          np.ndarray.flatten(prediction_test > 0.5))


print("Miou finished")


accuracy_test = accuracy_score(np.ndarray.flatten(np.array(Y_test, dtype=bool)),
                               np.ndarray.flatten(prediction_test > 0.5))



print("Accuracy finished")
