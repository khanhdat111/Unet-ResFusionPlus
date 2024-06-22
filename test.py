save_path = "/kaggle/working/best_model.weights.h5"
model = create_model(img_height=352, img_width=352, input_chanels=3, out_classes=1, starting_filters=34)
model.load_weights(save_path)
model.compile(optimizer=optimizer, loss=dice_metric_loss, metrics=["accuracy"])
