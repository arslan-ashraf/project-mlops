import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

print("=" * 100)
print("tensorflow version:", tf.__version__)

def get_data(args):

    print("=" * 100)
    print("Attempting to read the data ...")

    train_data_path = args.train_data_bucket + "/train.csv"
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


# scaling numerical values to [0, 1]
def get_rescaling_layers(columns_to_rescale, inputs_dict, x_train):

    print("columns_to_rescale:")
    print(columns_to_rescale)
    print("=" * 100)
    rescaling_layers = []

    for column in columns_to_rescale:
        input_layer = tf.keras.Input(shape=(1,), dtype='float32', name=column + "_input")
        rescaling_layer = tf.keras.layers.Rescaling(scale=1.0/x_train[column].max(), 
                                                    name=column + "_rescaling_layer")
        inputs_dict[column] = input_layer
        rescaling_layers.append(rescaling_layer(input_layer))

    print("Rescaling tensorflow layers:")
    print(rescaling_layers)
    print("=" * 100)

    return rescaling_layers


# standarding numerical values to have mean 0 and variance/standard deviation 1
def get_standardizing_layers(columns_to_standardize, inputs_dict, x_train):

    print("columns_to_standardize:")
    print(columns_to_standardize)
    print("=" * 100)

    standardizing_layers = []

    for column in columns_to_standardize:
        input_layer = tf.keras.Input(shape=(1,), dtype='float32', name=column + "_input")
        standardizing_layer = tf.keras.layers.Normalization(mean=x_train[column].mean(),
                                                            variance=x_train[column].var(),
                                                            axis=None,
                                                            name=column + "_standardizing_layer")
        inputs_dict[column] = input_layer
        standardizing_layers.append(standardizing_layer(input_layer))
    
    print("Standardizing tensorflow layers:")
    print(standardizing_layers)
    print("=" * 100)

    return standardizing_layers


# one hot encode all categorical columns
def get_one_hot_encoding_layers(columns_to_one_hot, inputs_dict, x_train):

    print("columns_to_one_hot:")
    print(columns_to_one_hot)
    print("=" * 100)
    one_hot_encoding_layers = []

    for column in columns_to_one_hot:
        input_layer = tf.keras.Input(shape=(1,), dtype='string', name=column + "_input")
        max_tokens = len(list(x_train[column].value_counts().keys())) + 1 # +1 is for unknown category
        vocabulary = list(x_train[column].value_counts().keys())
        one_hot_encoding_layer = tf.keras.layers.StringLookup(max_tokens=max_tokens, 
                                                              vocabulary=vocabulary,
                                                              output_mode="one_hot",
                                                              name=column + "_one_hot_encoding_layer")
        inputs_dict[column] = input_layer
        one_hot_encoding_layers.append(one_hot_encoding_layer(input_layer))
    
    print("One hot encoding tensorflow layers:")
    print(one_hot_encoding_layers)
    print("=" * 100)

    return one_hot_encoding_layers


# assemble the prepocessing layers inside a tensorflow model
# build a model that contains two models inside it, one preprocessing_model, 
# the other a trainable tensorflow model
def build_model(args, feature_engineering_dict, x_train):

    print("Building the model ...")
    inputs_dict = {}

    rescaling_layers = get_rescaling_layers(feature_engineering_dict["columns_to_rescale"], 
                                           inputs_dict, 
                                           x_train)

    standardizing_layers = get_standardizing_layers(feature_engineering_dict["columns_to_standardize"], 
                                                   inputs_dict,
                                                   x_train)

    one_hot_encoding_layers = get_one_hot_encoding_layers(feature_engineering_dict["columns_to_one_hot"], 
                                                         inputs_dict,
                                                         x_train)

    all_processing_layers = rescaling_layers + standardizing_layers + one_hot_encoding_layers

    concatenate_layer = tf.keras.layers.Concatenate(name="concatenate_layer")(all_processing_layers)

    best_hps_path = args.model_bucket + "/best_hyperparameters.csv"
    best_hps = pd.read_csv(best_hps_path)
    
    number_of_layers = int(best_hps["number_of_layers"].values[0])
    print("Number of layers:", number_of_layers)

    model_dense_layers = []
    for layer in range(1, number_of_layers + 1):
        layer_name = "dense_layer_" + str(layer)
        dense_layer = tf.keras.layers.Dense(units=200, activation='relu', name=layer_name)
        model_dense_layers.append(dense_layer)        
    
    scalar_output_layer = tf.keras.layers.Dense(units=1, name="output_layer")

    dense = model_dense_layers[0](concatenate_layer)
    for layer in model_dense_layers[1:]:
        dense = layer(dense)

    output_layer = scalar_output_layer(dense)

    model = tf.keras.Model(inputs=inputs_dict, outputs=output_layer, name="model")

    learning_rate = float(best_hps["learning_rate"].values[0])

    print("Learning rate:", learning_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=args.loss_function,
                  metrics=["mae", "mse"])

    print("Model built and compiled, model summary:")
    print(model.summary())
    print("=" * 100)

    return model


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

    print("Starting model training ...")
    model_history = model.fit(x_train_dict, 
                              y_train, 
                              validation_data=(x_valid_dict, y_valid),
                              epochs=args.epochs, 
                              batch_size=args.batch_size, 
                              callbacks=[tensorboard_callback])

    print("Model training finished")
    print("=" * 100)

    print(f"Attempting to save keras trainable model (with no model signatures) with timestamp {timestamp} ...")
    
    keras_model_save_path = args.model_bucket + f"/trainable_models/{timestamp}/model.keras"
    model.save(filepath=keras_model_save_path)

    print("Trainable model successfully saved at location:", keras_model_save_path)
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

    print("Attempting to save the model in the bucket:", args.model_bucket)

    model_artifacts_save_path = args.model_bucket + "/model_artifacts"
    
    tf.saved_model.save(obj=model,
                        export_dir=model_artifacts_save_path,
                        signatures=signatures)
    
    print("Model with serving function signature successfully saved")


def get_args():

    parser = argparse.ArgumentParser(description="Data preprocessing script")

    parser.add_argument("--train_data_bucket", type=str, required=True)

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

    # columns to apply different types of transformations
    columns_to_rescale = ["buyer_rating", "num_reviews", "percent_recommend", "year", "comfort_review", "interior_review",  "performance_review",   "value_review", "exterior_review", "reliability_review"]
    columns_to_standardize = ["cube_root_mileage", "average_mpg"]
    columns_to_one_hot = ["brand", "drivetrain", "accidents_or_damage", "clean_title", "one_owner_vehicle", "personal_use_only", "open_recall"]
    
    feature_engineering_dict = { "columns_to_rescale": columns_to_rescale,
                                 "columns_to_standardize": columns_to_standardize,
                                 "columns_to_one_hot": columns_to_one_hot}
    
    data_dict = { "x_train": x_train, "y_train": y_train,
                  "x_valid": x_valid, "y_valid": y_valid }

    model = build_model(args, feature_engineering_dict, x_train)

    train_model(args, model, data_dict)


if __name__ == "__main__":
    main()