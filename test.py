from sklearn.metrics import jaccard_score, accuracy_score, f1_score

from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture.Unet_ResFusionPlus import create_model
from ImageLoader.ImageLoader2D import load_data


save_path = "path_your_weights"
model = create_model(img_height=352, img_width=352, input_chanels=3, out_classes=1, starting_filters=34)
model.load_weights(save_path)
model.compile(optimizer=optimizer, loss=dice_metric_loss, metrics=["accuracy"])

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
