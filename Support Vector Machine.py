import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#np.random.seed(7)






df = pd.read_csv('data.csv')






Y = df['class'].values

X = df
del X['class']

del X['Unnamed: 0']

cols = X.columns

numeric = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']







emp = None

for j in cols:
    
	if emp is None:
        
		emp = pd.DataFrame(X[j], columns=[j])
    
	else:
        
		emp = emp.join(X[j])
    
	if not j in numeric:
        
		emp = pd.get_dummies(emp, columns=[j])

X = emp.values






encoder = LabelEncoder()

encoder.fit(Y)

Y = encoder.transform(Y)






X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)






model = SVC()

model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)
ypred = model.predict(X_test)
ytest = Y_test



print('output class: ', model.predict(X_test))
print ('Accuracy using Support Veator Machine: ', score*100, '%')

print(classification_report(ypred,ytest))
print(confusion_matrix(ytest, ypred))
