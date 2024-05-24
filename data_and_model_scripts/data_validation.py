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

def validate_schema_and_shape(df):

	print("Validating data schema, present columns, and shape ...")

	assert df.shape[1] == 25
	
	numerical_columns = ['buyer_rating', 'num_reviews', 'percent_recommend', 'year', 'comfort_review', 
                         'interior_review', 'performance_review', 'value_review', 'exterior_review', 
                         'reliability_review', 'mileage', 'average_mpg']
        
	string_columns = ['brand', 'drivetrain', 'accidents_or_damage', 'clean_title', 
					  'one_owner_vehicle', 'personal_use_only', 'open_recall']

	target_column = ['price']

	columns_to_drop = ["model", "exterior_color", "interior_color", "fuel_type", "transmission"]

	all_columns = numerical_columns + string_columns + target_column + columns_to_drop

	assert (np.sort(np.array(all_columns)) == np.sort(np.array(df.columns))).all()

	print("All expected columns are present and the shape is correct")

	numerical_columns_dtypes = np.array(df[numerical_columns + target_column].dtypes)
	
	assert ((numerical_columns_dtypes == 'float64') | (numerical_columns_dtypes == 'float32')).all()
	
	string_columns_dtypes = np.array(df[string_columns].dtypes)

	assert (string_columns_dtypes == 'object').all()

	print("All expected numerical and string columns are of appropriate data types")
	print("Schema, columns and shape validation successful")
	print("=" * 100)


def validate_all_rating_columns(df):

	print("Validating all rating columns, min: 1.0, max: 5.0, missing: -5.0")

	rating_columns = ['buyer_rating', 'comfort_review', 'interior_review', 
                  	  'performance_review', 'value_review', 'exterior_review', 
                  	  'reliability_review']

	assert df[rating_columns].values.min() == -5.0

	for column in rating_columns:
		column_without_missing = df[df[column] != -5.0][column]
		assert column_without_missing.min() == 1.0
		assert column_without_missing.max() == 5.0

	print("All rating columns successfully validated")
	print("=" * 100)


def validate_numerical_columns(df):

	print("Validating all numerical columns ...")
	print("Validating the price column ...")
	if df[df["price"] == 1750000.0].shape[0] == 1:
		extreme_price_index = df[df["price"] == 1750000.0].index
		df.drop(labels=extreme_price_index, inplace=True)

	assert ((df["price"] < 750000) | (df["price"] > 0)).all()
	print("Price validation successful")
	print("-" * 50)

	print("Validating the mileage column ...")
	if df[df["mileage"] == 1182285.0].shape[0] == 1:
		faulty_mileage_index = df[df["mileage"] == 1182285.0].index
		df.drop(labels=faulty_mileage_index, inplace=True)

	assert ((df["mileage"] <= 300000) | (df["mileage"] > 0)).all()
	print("Mileage validation successful")
	print("-" * 50)

	print("Validating average_mpg column ...")
	assert (df["average_mpg"].min() > 10) & (df["average_mpg"].max() < 100)
	print("Average mpg successfully validated")
	print("-" * 50)

	print("Validating percent_recommend column ...")
	assert df[df["percent_recommend"] != -100.0]["percent_recommend"].min() > 0
	assert df[df["percent_recommend"] != -100.0]["percent_recommend"].max() <= 100.0
	print("Percent of recommenders successful validated")
	print("-" * 50)

	print("Validating year column ...")
	assert df["year"].min() > 1985
	assert df["year"].max() < 2025
	print("Year column successfully validated")
	print("-" * 50)

	print("All numerical columns successfully validated")
	print("=" * 100)


def validate_string_columns(df):

	print("Validating all string columns ...")
	print("Validating accidents_or_damage column ...")
	accident_values = np.array(['At least 1 accident or damage reported', 
								'None reported'])
	assert (df["accidents_or_damage"].unique() == accident_values).all()
	print("Accident or damage column successfully validated")
	print("-" * 50)

	print("Validating drivetrain column ...")
	drivetrain_values = np.array(['Four-wheel Drive', 
								  'Rear-wheel Drive', 
								  'All-wheel Drive',
								  'Front-wheel Drive', 
								  'Unknown'])
	assert (df["drivetrain"].unique() == drivetrain_values).all()
	print("Drivetrain column successfully validated")
	print("-" * 50)

	print("Validating clean_title column ...")
	clean_title_values = np.array(['Unknown', ' Yes', ' No'])
	assert (df["clean_title"].unique() == clean_title_values).all()
	print("Clean title column successfully validated")
	print("-" * 50)

	print("Validating one_owner_vehicle column ...")
	one_owner_vehicle_values = np.array([' Yes', ' No', 'Unknown'])
	assert (df["one_owner_vehicle"].unique() == one_owner_vehicle_values).all()
	print("One owner vehicle column successfully validated")
	print("-" * 50)

	print("All string columns have been successfully validated")
	print("=" * 100)


def validate_data(df):

	validate_schema_and_shape(df)

	validate_all_rating_columns(df)

	validate_numerical_columns(df)

	validate_string_columns(df)

	print("Data validation step successful")


def get_args():

	parser = argparse.ArgumentParser(description="Data validation script")

	parser.add_argument("--data_bucket_uri", type=str, required=True)

	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()

	df = get_data(args)

	validate_data(df)