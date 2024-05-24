import pytest
import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
from data_prep_and_training.train import (get_data,
										  get_rescaling_layers,
										  get_standardizing_layers,
										  get_one_hot_encoding_layers,
										  build_model,
										  train_model)


def get_args():

    parser = argparse.ArgumentParser(description="Data preprocessing script")

    processed_data_bucket = "C:\\Users\\Arslan\\Desktop\\used-car-kubeflow-pipeline\\"
    train_data_path = processed_data_bucket + "train.csv"
    valid_data_path = processed_data_bucket + "valid.csv"

    parser.add_argument("--train_data_path", type=str, default=train_data_path)

    parser.add_argument("--valid_data_path", type=str, default=valid_data_path)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--loss_function", type=str, default="mae")

    model_save_bucket = "C:\\Users\\Arslan\\Desktop\\used-car-kubeflow-pipeline\\"
    model_artifacts_save_path = model_save_bucket + "model_artifacts"
    keras_model_save_path = model_save_bucket + "model.keras"

    parser.add_argument("--tensorboard_path", type=str, default=model_save_bucket)

    parser.add_argument("--model_artifacts_save_path", 
                        type=str, 
                        default=model_artifacts_save_path)

    parser.add_argument("--keras_model_save_path", type=str, default=keras_model_save_path)

    # only the first argument is necessary, the rest follow
    return parser.parse_args(["--train_data_path", train_data_path])


args = get_args()

def test_get_data():
	x_train, y_train, x_valid, y_valid = get_data(args)

	assert f"{type(x_train)}" == "<class 'pandas.core.frame.DataFrame'>"
	assert f"{type(y_train)}" == "<class 'pandas.core.series.Series'>"
	assert f"{type(x_valid)}" == "<class 'pandas.core.frame.DataFrame'>"
	assert f"{type(y_valid)}" == "<class 'pandas.core.series.Series'>"


x_train, y_train, x_valid, y_valid = get_data(args)

columns_to_rescale = ["buyer_rating", "num_reviews", "percent_recommend", "year", "comfort_review", "interior_review",  "performance_review",   "value_review", "exterior_review", "reliability_review"]
columns_to_standardize = ["cube_root_mileage", "average_mpg"]
columns_to_one_hot = ["brand", "drivetrain", "accidents_or_damage", "clean_title", "one_owner_vehicle", "personal_use_only", "open_recall"]

feature_engineering_dict = { "columns_to_rescale": columns_to_rescale,
                             "columns_to_standardize": columns_to_standardize,
                             "columns_to_one_hot": columns_to_one_hot}

def test_get_rescaling_layers():
	inputs_dict = {}
	columns_to_rescale = feature_engineering_dict["columns_to_rescale"]
	rescaling_layers = get_rescaling_layers(columns_to_rescale, inputs_dict, x_train)

	assert len(rescaling_layers) == len(columns_to_rescale)
	assert f"{type(rescaling_layers[0])}" == "<class 'keras.src.backend.common.keras_tensor.KerasTensor'>"


def test_get_standardizing_layers():
	inputs_dict = {}
	columns_to_standardize = feature_engineering_dict["columns_to_standardize"]
	standardizing_layers = get_standardizing_layers(columns_to_standardize, inputs_dict, x_train)

	assert len(standardizing_layers) == len(columns_to_standardize)
	assert f"{type(standardizing_layers[0])}" == "<class 'keras.src.backend.common.keras_tensor.KerasTensor'>"


def test_get_standardizing_layers():
	inputs_dict = {}
	columns_to_one_hot = feature_engineering_dict["columns_to_one_hot"]
	one_hot_encoding_layers = get_one_hot_encoding_layers(columns_to_one_hot, inputs_dict, x_train)
	assert len(one_hot_encoding_layers) == len(columns_to_one_hot)
	assert f"{type(one_hot_encoding_layers[0])}" == "<class 'keras.src.backend.common.keras_tensor.KerasTensor'>"


def test_build_model():
	model = build_model(args, feature_engineering_dict, x_train)

	example_input = { name: tf.convert_to_tensor(value[:1]) for name, value in x_train.items() }
	predictions = model(example_input)

	assert predictions.shape == (1, 1)

def get_column(matrix, index):
	return [matrix[i][index] for i in range(len(matrix))]

def test_train_model():
	if not os.path.exists(args.model_artifacts_save_path):
		data_dict = { "x_train": x_train, "y_train": y_train, 
					  "x_valid": x_valid, "y_valid": y_valid }
		model = build_model(args, feature_engineering_dict, x_train)
		train_model(args, model, data_dict)

	num_examples = 10

	test_float_inputs = [list(arr) for arr in x_train[:num_examples].values[:, :12]]
	test_string_inputs = [list(arr) for arr in x_train[:num_examples].values[:, 12:]]

	loaded_model = tf.saved_model.load(export_dir=args.model_artifacts_save_path)

	assert loaded_model.signatures["serving_default"] is not None

	model_forward_pass = loaded_model.signatures["serving_default"]
	
	predictions = model_forward_pass(float_inputs=test_float_inputs, 
									 string_inputs=test_string_inputs)

	name_of_predictions = list(model_forward_pass(
							float_inputs=test_float_inputs, 
							string_inputs=test_string_inputs).keys())[0]

	all_predictions = predictions[name_of_predictions]
	
	assert all_predictions.shape == (num_examples, 1)