import pandas as pd
import numpy as np
import argparse

pd.set_option("display.max_columns", None)


def get_data(args):

	print("=" * 100)
	print("Attempting to read the data ...")

	data_path = args.data_bucket_uri + "/cleaned_used_cars_data.csv"
	df = pd.read_csv(data_path)

	print("Read successful")
	print("df shape:", df.shape)
	print(type(df))
	print(df.head(3))
	print("=" * 100)

	return df


def remove_extreme_price_outlier(df):

	print("Remove extreme outlier in the price column")
	extreme_price_index = df[df["price"] == 1750000.0].index
	df.drop(labels=extreme_price_index, inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def remove_extreme_mileage_outlier(df):

	print("Remove one extreme (unrealistic) outlier in mileage column")
	faulty_mileage_index = df[df["mileage"] == 1182285.0].index
	df.drop(labels=faulty_mileage_index, inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def drop_low_value_columns(df):

	print("Drop little value columns: model, exterior_color, interior_color, fuel_type, transmission")
	df.drop(columns=["model", "exterior_color", "interior_color", "fuel_type", "transmission"], inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def apply_log_to_price(df):

	print("Transform the target column 'price' by applying the log function")
	df["price"] = df["price"].apply(np.log)
	df.rename(columns={ "price": "log_price" }, inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def apply_cube_root_to_mileage(df):

	print("Transform mileage column by applying the cube root function")
	df["mileage"] = df["mileage"].apply(np.cbrt)
	df.rename(columns={ "mileage": "cube_root_mileage" }, inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def impute_missing_mpg_with_median(df):

	print("Impute missing average_mpg with median value")
	print("median miles per gallon value:", df["average_mpg"].median())
	df["average_mpg"].fillna(df["average_mpg"].median(), inplace=True)
	print("df shape:", df.shape)
	print("=" * 100)


def rearrange_columns(df):

	print("Dataframe columns:")
	print(df.columns)
	print("Rearrange columns for easier feature engineering later")
	all_columns = list(df.columns)
	new_column_arrangement = all_columns[1:4] + [all_columns[5]] + all_columns[7:13] + [all_columns[0]] + [all_columns[14]] + [all_columns[6]] + [all_columns[13]] + all_columns[15:] + [all_columns[4]]
	df = df[new_column_arrangement]
	print("Columns after rearranging:")
	print(df.columns)
	print("=" * 100)
	return df


def split_data_and_save(args, df):

	print("Split the dataset in train, validation, and test sets")

	train_fraction = 1 - args.fraction_for_valid_and_test_data

	print(train_fraction, " fraction of data for training")
	train_df = df.sample(frac=train_fraction, random_state=0)
	valid_test_df = df.drop(labels=train_df.index)

	valid_df = valid_test_df.sample(frac=0.5, random_state=0)
	test_df = valid_test_df.drop(labels=valid_df.index)

	print("train dataset shape:", train_df.shape)
	print("valid dataset shape:", valid_df.shape)
	print("test dataset shape:", test_df.shape)
	print("=" * 100)

	print("Attempting to save the three datasets ...")
	print("Train, valid, test datasets save bucket:", args.processed_data_save_bucket)

	train_save_path = args.processed_data_save_bucket + "/train.csv"
	valid_save_path = args.processed_data_save_bucket + "/valid.csv"
	test_save_path = args.processed_data_save_bucket + "/test.csv"

	train_df.to_csv(train_save_path, index=False)
	valid_df.to_csv(valid_save_path, index=False)
	test_df.to_csv(test_save_path, index=False)
	print("Save successful")


def save_data_without_splitting(args, df):

	print("Attempting to save all data for training")
	print("Train dataset save bucket:", args.processed_data_save_bucket)

	train_save_path = args.processed_data_save_bucket + "/train.csv"
	df.to_csv(train_save_path, index=False)
	print("Save successful")


def process_data(args, df):	

	remove_extreme_price_outlier(df)

	remove_extreme_mileage_outlier(df)

	drop_low_value_columns(df)

	apply_log_to_price(df)

	apply_cube_root_to_mileage(df)

	impute_missing_mpg_with_median(df)

	df = rearrange_columns(df)

	if args.fraction_for_valid_and_test_data > 0.0:
		split_data_and_save(args, df)
	else:
		save_data_without_splitting(args, df)


def get_args():

	parser = argparse.ArgumentParser(description="Data preprocessing script")

	parser.add_argument("--data_bucket_uri", type=str, required=True)

	parser.add_argument("--fraction_for_valid_and_test_data", type=float, default=0.2)

	parser.add_argument("--processed_data_save_bucket", type=str, required=True)

	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()

	df = get_data(args)

	process_data(args, df)