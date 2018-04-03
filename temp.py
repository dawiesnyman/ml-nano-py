# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:46:36 2018
@author: Dawie
"""
#relationship_type	gender	smoker_status	BirthYear	CurrentAge	HasEmail	payment_frequency	premium_amount	PolicyAgeInMonths	CustomerStartAge	postcode

#compile renders package
#import py_compile
#py_compile.compile('render.py', 'renders.pyc')

#http://www.ritchieng.com/machine-learning-project-customer-segments/#
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from sklearn.preprocessing import Imputer
np.set_printoptions(threshold=np.nan)
#from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    #data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")
 
    
data.describe()    
    
data.loc[[100, 200, 300],:]    

x = data.iloc[:, :].values
y = data.iloc[:, 7].values

# Display a description of the dataset
data.describe()

# transform NaN values 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(x[:, 3:5])

x[:, 3:5] = imputer.transform(x[:, 3:5])

imputer.fit(x[:, 9:10])
x[:, 9:10] = imputer.transform(x[:, 9:10])



#x[:, 2] = x[:, 2].astype('category')

'''
x.types
pd.get_dummies(x[:, 0], prefix="relationship_type").head()

# test 
#x[2669]
'''
#handle categorical data
#columns 0, 1, 2, 5, 6
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_relationship = LabelEncoder()

x[:, 0] =le_relationship.fit_transform(x[:, 0])

x[:, 1] =le_relationship.fit_transform(x[:, 1])
x[:, 2] =le_relationship.fit_transform(x[:, 2])
x[:, 5] =le_relationship.fit_transform(x[:, 5])
x[:, 6] =le_relationship.fit_transform(x[:, 6])

#dummy encode relationship
ohe_relationship = OneHotEncoder(categorical_features=['smoker_status'])

z = ohe_relationship.fit_transform(x).toarray()
'''
a = pd.get_dummies(x[:, 0], prefix="relationship")
b = pd.get_dummies(x[:, 1], prefix="gender")
c = pd.get_dummies(x[:, 2], prefix="smoker_status")
d = pd.get_dummies(x[:, 5], prefix="has_email")
e = pd.get_dummies(x[:, 6], prefix="pay_freq")
f = pd.get_dummies(x[:, 10], prefix="postcode")

n = pd.concat([pd.DataFrame(x), a], axis=1)
n = pd.concat([pd.DataFrame(n), b], axis=1)
n = pd.concat([pd.DataFrame(n), c], axis=1)
n = pd.concat([pd.DataFrame(n), d], axis=1)
n = pd.concat([pd.DataFrame(n), e], axis=1)
n = pd.concat([pd.DataFrame(n), f], axis=1)


n.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(n)

kmeans.labels_array([0, 0, 0, 1, 1, 1], dtype=int32)

n.head()

n.drop([0], axis=1, inplace=True)
n.drop([1], axis=1, inplace=True)
n.drop([2], axis=1, inplace=True)
n.drop([5], axis=1, inplace=True)
n.drop([6], axis=1, inplace=True)
n.drop([10], axis=1, inplace=True)
x[2669]

z[2669]












































