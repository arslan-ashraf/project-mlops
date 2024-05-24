import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner
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


def concat_input_layers(args, feature_engineering_dict, x_train):

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

    return concatenate_layer


# assemble the prepocessing layers inside a tensorflow model
# build a model that contains two models inside it, one preprocessing_model, 
# the other a trainable tensorflow model
def build_model_with_hyperparameters(number_of_layers, learning_rate):

    print("Building the rest of the model with hyperparameters ...")
    print("Number of dense layers to build:", number_of_layers)
    print("The learning rate:", learning_rate)

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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=args.loss_function,
                  metrics=["mae", "mse"])

    print("Model built and compiled")
    print("=" * 100)

    return model


def build_model(hp):

    print("Initializing hyperparameters ...")
    number_of_layers = hp.Int(name="number_of_layers",
                              min_value=int(args.number_of_layers),
                              max_value=8,
                              step=2)

    learning_rate = hp.Float(name="learning_rate", 
                             min_value=args.learning_rate, 
                             max_value=0.1, 
                             sampling="log")

    print("Number of layers is:", number_of_layers)
    print("Learning rate is:", learning_rate)
    print("=" * 100)

    model = build_model_with_hyperparameters(number_of_layers=number_of_layers,
                                             learning_rate=learning_rate)

    return model


def run_hp_tuner(args, data_dict):

    x_train = data_dict["x_train"]
    y_train = data_dict["y_train"]

    x_valid = data_dict["x_valid"]
    y_valid = data_dict["y_valid"]

    x_train_dict = { name: tf.convert_to_tensor(value) for name, value in x_train.items() }
    x_valid_dict = { name: tf.convert_to_tensor(value) for name, value in x_valid.items() }

    print("Initializing keras tuner ...")

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=int(args.max_trials),
        overwrite=True,
        directory="tuner_dir",
        project_name="used-cars-hp-tuning"
    )

    print("Tuner search space summary:")
    print(tuner.search_space_summary())

    print("Starting hyperparameter search ...")
    model_history = tuner.search(x_train_dict, 
                                 y_train, 
                                 epochs=args.epochs,
                                 validation_data=(x_valid_dict, y_valid),
                                 batch_size=int(args.batch_size))

    print("HyperParameter search finished")
    print("=" * 100)

    print("Best model summary:")
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
    best_model.summary()

    print("=" * 100)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters found:")
    print(best_hps.values)
    print("Best number_of_layers:", best_hps.values["number_of_layers"])
    print("Best learning_rate:", best_hps.values["learning_rate"])
    print("=" * 100)

    print("Writing the best hyperparameters to a pandas dataframe ...")

    best_hps_df = pd.DataFrame.from_dict(data=best_hps.values, orient="index").transpose()

    best_hps_save_path = args.model_bucket + "/best_hyperparameters.csv"

    print("best_hps dataframe:")
    print(best_hps_df)
    
    best_hps_df.to_csv(best_hps_save_path, index=False)

    print("Successfully written the best hyperparameters to the path:", best_hps_save_path)


def get_args():

    parser = argparse.ArgumentParser(description="Data preprocessing script")

    parser.add_argument("--train_data_bucket", type=str, required=True)

    parser.add_argument("--valid_data_bucket", type=str, required=True)

    parser.add_argument("--model_bucket", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--loss_function", type=str, default="mse")

    parser.add_argument("--max_trials", type=int, default=15)

    # hyper parameters to tune
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    parser.add_argument("--number_of_layers", type=float, default=4)

    return parser.parse_args()


def main():

    global args
    args = get_args()

    x_train, y_train, x_valid, y_valid = get_data(args)

    # columns to apply different types of transformations
    columns_to_rescale = ["buyer_rating", "num_reviews", "percent_recommend", "year", "comfort_review", "interior_review",  "performance_review",   "value_review", "exterior_review", "reliability_review"]
    columns_to_standardize = ["cube_root_mileage", "average_mpg"]
    columns_to_one_hot = ["brand", "drivetrain", "accidents_or_damage", "clean_title", "one_owner_vehicle", "personal_use_only", "open_recall"]
    
    global feature_engineering_dict
    feature_engineering_dict = { "columns_to_rescale": columns_to_rescale,
                                 "columns_to_standardize": columns_to_standardize,
                                 "columns_to_one_hot": columns_to_one_hot}
    
    global inputs_dict
    inputs_dict = {}

    global concatenate_layer
    concatenate_layer = concat_input_layers(args, feature_engineering_dict, x_train)

    data_dict = { "x_train": x_train, "y_train": y_train,
                  "x_valid": x_valid, "y_valid": y_valid }

    run_hp_tuner(args, data_dict)


if __name__ == "__main__":
    main()