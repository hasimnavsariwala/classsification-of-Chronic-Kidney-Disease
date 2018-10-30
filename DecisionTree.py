import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import graphviz
import sklearn.datasets as dtst


# In[2]:


#np.random.seed(4)


df = pd.read_csv('data.csv')



# In[3]:



Y = df['class'].values

X = df
del X['class']

del X['Unnamed: 0']

cols = X.columns

numeric = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']




for i in numeric:
	d = pd.DataFrame(X[i], columns=[i])
X = d.values

# In[4]:


#emp = None

#for j in cols:
    
#	if emp is None:
        
#		emp = pd.DataFrame(X[j], columns=[j])
    
#	else:
        
#		emp = emp.join(X[j])
    
#	if not j in numeric:
        
#		emp = pd.get_dummies(emp, columns=[j])
#X = emp.values







# In[5]:



encoder = LabelEncoder()

encoder.fit(Y)

Y = encoder.transform(Y)



# In[6]:



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)



# In[7]:



model = DecisionTreeClassifier()

model.fit(X_train,Y_train)

score = model.score(X_test, Y_test)


ypred = model.predict(X_test)
ytest = Y_test

print('Accuracy using Decision Tree Classifier: ', score*100, '%')

print(classification_report(ypred,ytest))
print(confusion_matrix(ypred,ytest))

from sklearn.datasets import load_iris
iris = load_iris()

model = model.fit(iris.data,iris.target)
 
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
#import pydotplus

#dot_data = stringIO()
dot_data = export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('Iris') 


graph = graphviz.Source(dot_data)  
graph  