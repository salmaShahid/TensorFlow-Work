import pandas as pd
census = pd.read_csv(r'C:\Users\LeNoVo T430\PycharmProjects\MlLib\Tensorflow\census_data_Classification.csv')
print(census.head())
#TensorFlow won't be able to understand strings as labels, you'll need to use pandas .apply() method to apply a custom function that converts them to 0s and 1s.
#Convert the Label column to 0s and 1s instead of strings.
census['income_bracket'].unique()
def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1

census['income_bracket'] = census['income_bracket'].apply(label_fix)

#Perform a Train Test Split on the Data
from sklearn.model_selection import train_test_split
x_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=1)

#Create the Feature Columns for tf.esitmator
print(census.columns)
import tensorflow as tf
#Create the tf.feature_columns for the categorical values. Use vocabulary lists or just use hash buckets.
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#Create the continuous feature_columns for the continuous values using numeric_column
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


#Put all these variables into a single list with the variable name feat_cols
feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]

#Create Input Function
#Batch_size is up to me (shuffle!)

input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,
                                               y=y_train,
                                               batch_size=100,
                                               num_epochs=None,
                                               shuffle=True)

#Create model with tf.estimator
#Create a LinearClassifier.(For DNNClassifier,
#  we'll need to create embedded columns out of the
# cateogrical feature that use strings)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

#train your model on the data, for at least 5000 steps.
model.train(input_fn=input_func,steps=5000)

#Evaluation Create a prediction input function. Remember to only supprt X_test data and keep shuffle=False.
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)

#Use model.predict() and pass in your input function. This will produce a generator of predictions,
# which we can then transform into a list, with list()
predictions = list(model.predict(input_fn=pred_fn))

print(predictions[0])
#Create a list of only the class_ids key values from the prediction list of dictionaries,
# these are the predictions we will use to compare against the real y_test values.
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(final_preds[:10])

#Import classification_report from sklearn.metrics and then see if we can figure out how to use it
#  to easily get a full report of  model's performance on the test data.
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=len(X_test),shuffle=False)

print(classification_report(y_test,final_preds))
results = model.evaluate(eval_input_func)
print(results)