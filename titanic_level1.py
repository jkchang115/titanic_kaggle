"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""
from util import *
import math
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	############################
	is_header = True

	with open(filename, 'r') as file:
		for line in file:
			if is_header:
				headers = line.strip().split(',')
				if mode == 'Train':
					headers = [headers[i] for i in range(len(headers)) if i not in {0, 3, 8, 10}]
				else:
					headers = [headers[i] for i in range(len(headers)) if i not in {0, 2, 7, 9}]
				# Initialize the keys in the data dictionary
				for header in headers:
					if header not in data:
						data[header] = []
				is_header = False
			else:
				values = line.strip().split(',')
				if mode == 'Train':
					values = [values[i] for i in range(len(values)) if i not in {0, 3, 4, 9, 11}]
					if '' not in values:
						for idx, header in enumerate(headers):
							if idx in {0, 1, 4, 5}:
								data[header].append(int(values[idx]))
							elif idx == 2:
								data[header].append(1 if values[idx] == 'male' else 0)
							elif idx in {3, 6}:
								data[header].append(float(values[idx]))
							elif idx == 7:
								data[header].append({'S': 0, 'C': 1, 'Q': 2}[values[idx]])
				else:
					values = [values[i] for i in range(len(values)) if i not in {0, 2, 3, 8, 10}]
					for idx, header in enumerate(headers):
						if idx in {0, 3, 4}:
							data[header].append(int(values[idx]))
						elif idx == 1:
							data[header].append(1 if values[idx] == 'male' else 0)
						elif idx == 2:
							data[header].append(float(values[idx]) if values[idx] else round(
								sum(training_data['Age']) / len(training_data['Age']), 3))
						elif idx == 5:
							data[header].append(float(values[idx]) if values[idx] else round(
								sum(training_data['Fare']) / len(training_data['Fare']), 3))
						elif idx == 6:
							data[header].append({'S': 0, 'C': 1, 'Q': 2}[values[idx]])

	return data
def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	if feature == 'Pclass':
		class_mapping = {1: 'Pclass_0', 2: 'Pclass_1', 3: 'Pclass_2'}
		for class_name in class_mapping.values():
			data[class_name] = []

		for value in data['Pclass']:
			for class_value, class_name in class_mapping.items():
				data[class_name].append(1 if value == class_value else 0)

	# General handling for categorical features like 'Sex' or 'Embarked'
	else:
		unique_values = list(set(data[feature]))  # Find unique values in the feature
		for unique_value in unique_values:
			new_column = f"{feature}_{unique_value}"
			data[new_column] = [1 if val == unique_value else 0 for val in data[feature]]

	# Remove the original column
	if feature in data:
		data.pop(feature)

	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	for feature in data:
		min_val = min(data[feature])
		max_val = max(data[feature])
		range_val = max_val - min_val

		# Apply normalization if range is not zero to avoid division by zero
		if range_val > 0:
			data[feature] = [(val - min_val) / range_val for val in data[feature]]
		else:
			# If all values are the same, they will be set to 0
			data[feature] = [0] * len(data[feature])

	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Start training
	for epoch in range(num_epochs):
		for i in range(len(labels)):
			# Step 3 : Feature Extract
			individual_data = {key: inputs[key][i] for key in keys}

			if degree == 2:
				for j in range(len(keys)):
					for k in range(j, len(keys)):
						individual_data[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]

			# Calculate the score of each feature
			score = dotProduct(individual_data, weights)

			# Apply the sigmoid function to the score for logistic regression probability
			h = 1 / (1 + math.exp(-score))

			# Step 4 : Update weights
			increment(weights, -alpha*(h-labels[i]), individual_data)

	return weights
