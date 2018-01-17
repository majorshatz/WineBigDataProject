#Used https://www.youtube.com/watch?v=0cGOwjmYL4s as a template
# coding: utf-8




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#read data
dataStore=pd.read_csv('winequality-whiteV3.csv')




#Machine learning part time to train data
from sklearn.cross_validation import train_test_split

X=dataStore.drop('qualities',axis=1)
y=dataStore['qualities']
#Used method from from sklearn cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#start with one decision tree

from sklearn.tree import DecisionTreeClassifier
duhTree= DecisionTreeClassifier()
duhTree.fit(X_train,y_train)

#predict how well tree used in classification reports
predictions=duhTree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#Classification tree of one tree
print("Classification report")
print(classification_report(y_test,predictions))



#now try random forest
from sklearn.ensemble import RandomForestClassifier



randoForest=RandomForestClassifier(n_estimators=700)


randoForest.fit(X_train,y_train)


randoForest_pred=randoForest.predict(X_test)



print("Classification report")
print(classification_report(y_test,randoForest_pred))



#number of qualities(more numbers of 6 and 5 could effect data)
dataStore['qualities'].value_counts()



from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(dataStore.columns[1:])
features



#full tree to complex for interpretation 
dot_data = StringIO()  
export_graphviz(duhTree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


#full tree to complex for interpretation 
dot_data = StringIO()  
export_graphviz(duhTree, out_file=dot_data,feature_names=features,filled=True,rounded=True,max_depth=4)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  

