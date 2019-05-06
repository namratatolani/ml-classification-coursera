import pandas as pd
import string
import json
import numpy as np

data = pd.read_csv("../data/amazon_baby_subset.csv")
print("** fill na in review column **")
data.review.fillna('', inplace=True)

print("** remove punctuations **")
def remove_punctuation(text):
	text = text.translate(str.maketrans('','',string.punctuation))
	return text
data["review_clean"] = data["review"].apply(remove_punctuation)

print("** load important words **")
with open("../data/important_words.json") as f:
	important_words = json.load(f)

print("** each word becomes column with frequency as value **")
for word in important_words:
	data[word] = data["review_clean"].apply(lambda s : s.split().count(word))

print("** convert features (contant 1 + important_words) into matrices **")
def get_numpy_matrix_data(data, feature_columns, label_column):
	data["intercept"] = 1
	feature_columns = ["intercept"] + feature_columns
	features_frame = data[feature_columns]
	features_matrix = features_frame.as_matrix()
	labels = data[label_column]
	label_array = labels.as_matrix()
	return (features_matrix, label_array)
feature_matrix, sentiment = get_numpy_matrix_data(data, important_words, 'sentiment')
print("feature matrix shape")
print(feature_matrix.shape)

print("** predict probability through link function **")
def predict_probability(feature_matrix, coefficients):
	score = feature_matrix.dot(coefficients)
	predictions = 1/(1+np.exp(-score))
	return predictions

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print ("** checkpoint **")
print('correct_predictions           =', correct_predictions)
print('output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients))

print("** Compute derivative of log likelihood with respect to a single coefficient **")
def feature_derivative(errors, feature):
	derivative = sum(feature * errors)
	return derivative

print("** taking gradient steps **")
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
	coefficients = np.array(initial_coefficients)
	print("** for max_iterations calculate error **")
	for itr in range(max_iter):
		predictions = predict_probability(feature_matrix, coefficients)
		indicator = (sentiment==+1)
		errors = indicator - predictions
		print("** find derivative for itr "+str(itr) +" ** ")
		for j in range(len(coefficients)):
			derivative = feature_derivative(errors, feature_matrix[:,j])
			coefficients[j] = coefficients[j] + (step_size * derivative)
	return coefficients

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194), step_size=1e-7, max_iter=301)

print("** compute scores with final coefficients **")
scores = np.dot(feature_matrix, coefficients)

print("** final predictions **")
def classPredictions(score):
	if score > 0:
		return 1
	else:
		return -1
pred_func = np.vectorize(classPredictions)
predictions = pred_func(scores)

print("** number of reviews with positive sentiments **" + str(predictions[predictions == 1].sum()))

print("** measuring accuracy **")
correct_pred = (sentiment == predictions).sum()
accuracy = correct_pred / len(data)
print("accuracy = %.2f" % accuracy)


		
	
		











