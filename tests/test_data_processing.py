import pytest
import pandas as pd
import numpy as np
import argparse
import os
from data_prep_and_training.data_processing import (get_data,
													remove_extreme_price_outlier, 
													remove_extreme_mileage_outlier,
													drop_low_value_columns,
													apply_log_to_price,
													apply_cube_root_to_mileage,
													impute_missing_mpg_with_median,
													rearrange_columns,
													split_data_and_save,
													process_data)

def get_args():

	parser = argparse.ArgumentParser(description="Data preprocessing script")

	data_bucket = "C:\\Users\\Arslan\\Downloads\\"
	data_path = data_bucket + "cleaned_used_cars_data.csv"

	parser.add_argument("--data_path", type=str, default=data_path)

	processed_data_bucket = "C:\\Users\\Arslan\\Desktop\\used-car-kubeflow-pipeline\\"
	train_save_path = processed_data_bucket + "train.csv"
	valid_save_path = processed_data_bucket + "valid.csv"
	test_save_path = processed_data_bucket + "test.csv"

	parser.add_argument("--train_save_path", type=str, default=train_save_path)

	parser.add_argument("--valid_save_path", type=str, default=valid_save_path)

	parser.add_argument("--test_save_path", type=str, default=test_save_path)

	# only the first argument is necessary, the rest follow
	return parser.parse_args(["--data_path", data_path])


args = get_args()


def test_get_data():
	df = get_data(args)

	assert f"{type(df)}" == "<class 'pandas.core.frame.DataFrame'>"


df = get_data(args)

def test_remove_extreme_price_outlier():
	remove_extreme_price_outlier(df)
	new_df = df[df["price"] >= 1750000]

	assert new_df.shape[0] == 0


def test_remove_extreme_mileage_outlier():
	remove_extreme_mileage_outlier(df)
	new_df = df[df["mileage"] >= 1182285]

	assert new_df.shape[0] == 0


def test_drop_low_value_columns():
	original_num_columns = df.shape[1]

	drop_low_value_columns(df)
	new_num_columns = df.shape[1]

	assert (original_num_columns - 5) == new_num_columns


def test_apply_log_to_price():
	original_price = df["price"]
	
	apply_log_to_price(df)
	new_price = df["log_price"]

	assert (np.log(original_price) == new_price).all()


def test_apply_cube_root_to_mileage():
	original_mileage = df["mileage"]
	
	apply_cube_root_to_mileage(df)
	new_mileage = df["cube_root_mileage"]

	assert (np.cbrt(original_mileage) == new_mileage).all()


def test_impute_missing_mpg_with_median():
	df = get_data(args)
	missing_indices = df[df["average_mpg"].isnull()].index
	num_missing_before = df["average_mpg"].isnull().sum(axis=0)
	median_mpg = df["average_mpg"].median()

	impute_missing_mpg_with_median(df)

	num_missing_after = df["average_mpg"].isnull().sum(axis=0)

	assert num_missing_before > 0
	assert num_missing_after == 0
	assert (df.iloc[missing_indices]["average_mpg"] == median_mpg).all()


def test_rearrange_columns():
	df = get_data(args)
	columns_before = df.columns

	new_df = rearrange_columns(df)
	columns_after = new_df.columns

	assert np.array_equal(columns_before, columns_after) == False
	assert columns_after[-1] == "price"


def test_split_data_and_save():
	if not os.path.exists(args.train_save_path):
		split_data_and_save(args, df)

	train_df = pd.read_csv(args.train_save_path)

	assert train_df.shape[0] < df.shape[0]