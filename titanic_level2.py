"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

# Dictionary to keep track of the mean values for handling missing data during testing
mean_values = {}
def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None
	# Remove unnecessary columns from the dataset
	columns_to_remove = ['PassengerId', 'Name', 'Ticket', 'Cabin']
	data.drop(columns=columns_to_remove, inplace=True)

	# Convert categorical string values to numeric representations
	data['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
	data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

	if mode == 'Train':
		# Eliminate rows with any missing values
		data.dropna(inplace=True)

		# Store the mean of 'Age', 'Embarked', and 'Fare' to handle missing values in the test set
		global mean_values
		mean_values['Age'] = data['Age'].mean().round(3)
		mean_values['Embarked'] = data['Embarked'].mode()[0].round(3)
		mean_values['Fare'] = data['Fare'].mean().round(3)

		# Separate out the label data from the features
		labels = data.pop('Survived')

	elif mode == 'Test':
		# Fill in missing values in the test data using the means from the training data
		data['Age'].fillna(mean_values['Age'], inplace=True)
		data['Embarked'].fillna(mean_values['Embarked'], inplace=True)
		data['Fare'].fillna(mean_values['Fare'], inplace=True)

	# Return the appropriate data based on the mode
	if mode == 'Train':
		return data, labels
	else:
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	# Perform one-hot encoding on the specified feature
	one_hot = pd.get_dummies(data[feature], prefix=feature)

	# Ensure that all possible columns are created
	if feature == 'Embarked':
		for i in range(3):
			col_name = f'{feature}_{i}'
			if col_name not in one_hot.columns:
				one_hot[col_name] = 0

	if feature == 'Pclass':
		for i in range(3):
			col_name = f'{feature}_{i}'
			if col_name not in one_hot.columns:
				one_hot[col_name] = 0

	# Concatenate the new one-hot encoded columns to the original dataframe
	data = pd.concat([data, one_hot], axis=1)

	# Drop the original column as it is now replaced by the one-hot encoded columns
	data.drop(columns=[feature], inplace=True)

	return data

def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	# Create a scaler for each column and standardize the data
	features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
	extracted_features = data[features]
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(extracted_features)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83426966
	TODO: real accuracy on degree3 -> 0.86376404
	"""
	# Data pre-processing
	training_data, labels = data_preprocess(TRAIN_FILE, mode='Train')

	# One hot encoding
	features_to_encode = ['Sex', 'Pclass', 'Embarked']
	for feature in features_to_encode:
		training_data = one_hot_encoding(training_data, feature)

	# Standardization
	features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_0', 'Sex_1', 'Pclass_0','Pclass_1', 'Pclass_2', 'Embarked_0', 'Embarked_1']
	x_train = training_data[features]
	standardizer = preprocessing.StandardScaler()
	x_train= standardizer.fit_transform(x_train)

	# Create polynomial features and train the model for different degrees
	for degree in range(1, 4):
		poly_phi = preprocessing.PolynomialFeatures(degree=degree)
		x_train_poly = poly_phi.fit_transform(x_train)

		# Training
		h = linear_model.LogisticRegression(max_iter=10000)
		classifier = h.fit(x_train_poly, labels)
		acc = round(classifier.score(x_train_poly, labels), 8)
		print(f'Accuracy on degree{degree}: {acc:.8f}')


if __name__ == '__main__':
	main()
