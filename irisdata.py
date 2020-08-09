import pandas as pd
irisdata = pd.read_csv('iris_csv.csv')
print(irisdata.head())

X = irisdata.iloc[:,0:4]
y = irisdata.select_dtypes(include=[object])
print(y.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),max_iter=1000)
mlp.fit(X_train,y_train.values.ravel())
predictions = mlp.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,predictions)
a=a*100
print("Accuracy:",a,"%")