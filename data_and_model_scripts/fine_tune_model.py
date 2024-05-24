import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
import warnings
import os
from google.cloud import storage

# os.system("pip install google-cloud-storage")

warnings.filterwarnings("ignore")

print("=" * 100)
print("tensorflow version:", tf.__version__)

def get_data(args):

    print("=" * 100)
    print("Attempting to read the data ...")

    train_data_path = args.new_train_data_bucket + "/train.csv"
    valid_data_path = args.valid_data_bucket + "/valid.csv"

    x_train = pd.read_csv(train_data_path)
    x_valid = pd.read_csv(valid_data_path)

    print("Read successful")
    print("x_train shape:", x_train.shape)
    print("x_valid shape:", x_valid.shape)
    print(x_train.head(3))
    print("=" * 100)

    y_train = x_train.pop("log_price")
    y_valid = x_valid.pop("log_price")

    return x_train, y_train, x_valid, y_valid


def get_model(args):

    print("Attempting to load the saved model from model_bucket ...")

    gcs = storage.Client()

    # remove the "gs://" part from the bucket uri
    model_bucket_oject = gcs.bucket(args.model_bucket[5:])
    print("model_bucket_oject:", model_bucket_oject)

    model_folders = gcs.list_blobs(model_bucket_oject, 
                                   prefix="trainable_models")

    print("model_bucket_oject blobs object:", model_folders)
    timestamps_set = set()

    for folder in model_folders: 
        timestamp = folder.name.split("/")[1]
        if len(timestamp) > 0:
            timestamps_set.add(timestamp)

    print("timestamps_set:", timestamps_set)
    all_timestamps = sorted(timestamps_set, reverse=True)
    latest_timestamp = all_timestamps[0]
    
    latest_saved_model_path = args.model_bucket + "/trainable_models/" + latest_timestamp + "/model.keras"

    print("latest_saved_model_path:", latest_saved_model_path)

    loaded_model = tf.keras.models.load_model(filepath=latest_saved_model_path)

    print("Saved model successfully loaded")
    print("=" * 100)

    return loaded_model


def train_model(args, model, data_dict):

    x_train = data_dict["x_train"]
    y_train = data_dict["y_train"]

    x_valid = data_dict["x_valid"]
    y_valid = data_dict["y_valid"]

    x_train_dict = { name: tf.convert_to_tensor(value) for name, value in x_train.items() }
    x_valid_dict = { name: tf.convert_to_tensor(value) for name, value in x_valid.items() }

    # date_and_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    timestamp = str(int(time.time()))
    tensorboard_logs_dir = args.model_bucket + "/tensorboard_logs-" + timestamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_dir, histogram_freq=1)

    print("Starting model fine tuning ...")
    model_history = model.fit(x_train_dict, 
                              y_train, 
                              validation_data=(x_valid_dict, y_valid),
                              epochs=args.epochs, 
                              batch_size=args.batch_size, 
                              callbacks=[tensorboard_callback])

    print("Model fine tuning finished")
    print("=" * 100)

    print(f"Attempting to save fine tuned keras trainable model (with no model signatures) with timestamp {timestamp} ...")
    
    keras_model_save_path = args.model_bucket + f"/trainable_models/{timestamp}/model.keras"

    print("keras_model_save_path:", keras_model_save_path)

    model.save(filepath=keras_model_save_path)

    print("Trainable keras model successfully saved at location:", keras_model_save_path)
    print("=" * 100)

    @tf.function
    def preprocessing_func(float_inputs, string_inputs):
        numerical_columns = ['buyer_rating', 'num_reviews', 'percent_recommend', 'year', 'comfort_review', 
                             'interior_review', 'performance_review', 'value_review', 'exterior_review', 
                             'reliability_review', 'cube_root_mileage', 'average_mpg']
        
        string_columns = ['brand', 'drivetrain', 'accidents_or_damage', 'clean_title', 'one_owner_vehicle', 
                          'personal_use_only', 'open_recall']
        
        processed_inputs = {}
        for i, column in enumerate(numerical_columns):
            processed_inputs[column] = tf.convert_to_tensor(float_inputs[:, i], dtype=tf.float32)

        for i, column in enumerate(string_columns):
            processed_inputs[column] = tf.convert_to_tensor(string_inputs[:, i], dtype=tf.string)

        return processed_inputs

    @tf.function
    def post_processing_func(raw_predictions):
        predictions = tf.math.exp(raw_predictions, name="predictions")
        return predictions

    @tf.function
    def model_serving_function(float_inputs, string_inputs):
        processed_inputs = preprocessing_func(float_inputs, string_inputs)
        raw_predictions = model(processed_inputs)
        predictions = post_processing_func(raw_predictions)
        return predictions

    serving_function = model_serving_function.get_concrete_function(
        float_inputs=tf.TensorSpec(shape=[None, 12], dtype=tf.float32, name="float_inputs"),
        string_inputs=tf.TensorSpec(shape=[None, 7], dtype=tf.string, name="string_inputs")
    )

    signatures = { "serving_default": serving_function }

    print("Attempting to save the model with signatures in the bucket:", args.model_bucket)

    model_artifacts_save_path = args.model_bucket + "/model_artifacts"
    
    tf.saved_model.save(obj=model,
                        export_dir=model_artifacts_save_path,
                        signatures=signatures)
    
    print("Model with serving function signature successfully saved")
    print("=" * 100)


def get_args():

    parser = argparse.ArgumentParser(description="Data preprocessing script")

    parser.add_argument("--new_train_data_bucket", type=str, required=True)

    parser.add_argument("--valid_data_bucket", type=str, required=True)

    parser.add_argument("--model_bucket", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--loss_function", type=str, default="mse")

    parser.add_argument("--learning_rate", type=float, default=0.001)

    return parser.parse_args()


def main():
    args = get_args()

    x_train, y_train, x_valid, y_valid = get_data(args)

    model = get_model(args)
    
    data_dict = { "x_train": x_train, "y_train": y_train,
                  "x_valid": x_valid, "y_valid": y_valid }

    train_model(args, model, data_dict)


if __name__ == "__main__":
    main()