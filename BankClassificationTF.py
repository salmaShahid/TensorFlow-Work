import pandas as pd
data = pd.read_csv('BankNote_Authentication.csv')
# print(data)

import seaborn as sns
print(sns.countplot(x='class',data=data))
print(sns.pairplot(data,hue='class'))

#prepare data
from sklearn.preprocessing import StandardScaler
#create scaler object
scaler = StandardScaler()
#fir scaler to feature
scaler.fit(data.drop('class',axis=1))
#transform feature to scale version
scaled_features = scaler.fit_transform(data.drop('class',axis=1))
#convert sacle feature to dataframe
Data_feature = pd.DataFrame(scaled_features,columns=data.columns[:-1]) #without class attribute
print(Data_feature.head())
#traintest split
x = Data_feature
y = data['class']
#convert it into array in order to tf accept it as arrays not pandas series daata
x = x.as_matrix()
y = y.as_matrix()

#skleran traintest split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#contrib learn by tf
# import tensorflow
# import tensorflow.contrib.learn as learn
# #object classifier which is dnn use as learn and set it to have 2 classes and a[10,20,10] hidden unit layer structure
# feature_columns = [tensorflow.contrib.layers.real_valued_column("", dimension=1)]
# classifier = learn.DNNClassifier(hidden_units=[10,20,10],n_classes=2,feature_columns=feature_columns)
# classifier.fit(X_train,Y_train,steps=200,batch_size=20)
#
# #model evaluation use predict method for predictions
# note_predictions = classifier.predict(X_test)
# #classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(Y_test,note_predictions))
# print(classification_report(Y_test,note_predictions))

#compare it with random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,Y_train)
rfc_predict = rfc.predict(X_test)
print(confusion_matrix(Y_test,rfc_predict))
print(classification_report(Y_test,rfc_predict))

