#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[2]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object
# 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[ ]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[7]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[8]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[9]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[10]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[11]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[12]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[13]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[14]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[15]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[16]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[17]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[18]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# # K Nearest Neighbor(KNN)
# 
# 
# 

# In[116]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[117]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 30
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[118]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[119]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[120]:


k = 7
neigh7 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat7 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))


# In[121]:


from sklearn.metrics import log_loss
y_hat_probKNN = neigh7.predict_proba(X_test)
logKNN = log_loss(y_test, y_hat_probKNN)


# In[160]:


from sklearn.metrics import f1_score
f1KNN = f1_score(y_test, yhat7, average='weighted') 


# In[123]:


from sklearn.metrics import jaccard_score
jacKNN = jaccard_score(y_test, yhat7, pos_label='PAIDOFF')


# # Decision Tree
# 

# In[124]:


import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loanTree # it shows the default parameters


# In[126]:


loanTree.fit(X_train,y_train)


# In[127]:


predTree = loanTree.predict(X_test)


# In[128]:


#visual predictions of the values
print (predTree [0:5])
print (y_test [0:5])


# In[129]:


#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[130]:


tree.plot_tree(loanTree)
plt.show()


# In[131]:


from sklearn.metrics import log_loss
y_hat_probTREE = loanTree.predict_proba(X_test)
logTREE = log_loss(y_test, y_hat_probTREE)


# In[162]:


from sklearn.metrics import f1_score
f1TREE = f1_score(y_test, predTree, average='weighted') 


# In[134]:


from sklearn.metrics import jaccard_score
jacTREE = jaccard_score(y_test, predTree, pos_label='PAIDOFF')


# # Support Vector Machine
# 

# In[135]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[136]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[137]:


yhat = clf.predict(X_test)
yhat [0:5]


# In[138]:


#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("SVM Accuracy rbf: ", metrics.accuracy_score(y_test, yhat))


# In[139]:


clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)
yhat [0:5]
#evaluation

print("SVM Accuracy sigmoid: ", metrics.accuracy_score(y_test, yhat))


# In[140]:


clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)
yhat [0:5]
#evaluation

print("SVM Accuracy polynomial: ", metrics.accuracy_score(y_test, yhat))


# In[141]:


clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)
yhat [0:5]
#evaluation

print("SVM Accuracy linear: ", metrics.accuracy_score(y_test, yhat))


# In[142]:


from sklearn.metrics import f1_score
f1SVM = f1_score(y_test, yhat, average='weighted') 


# In[143]:


from sklearn.metrics import jaccard_score
jacSVM = jaccard_score(y_test, yhat, pos_label='PAIDOFF')


# In[144]:


from sklearn.metrics import log_loss
y_hat_probSVM = clf.predict_proba(X_test)
logSVM = log_loss(y_test, y_hat_probSVM)


# # Logistic Regression
# 

# In[145]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[146]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[147]:


y_hat = LR.predict(X_test)
y_hat


# In[148]:


y_hat_prob = LR.predict_proba(X_test)



# In[149]:


from sklearn.metrics import jaccard_score
jacLOG = jaccard_score(y_test, y_hat,pos_label='PAIDOFF')


# In[151]:



from sklearn.metrics import f1_score
f1LOG = f1_score(y_test, y_hat, average='weighted') 


# In[152]:


from sklearn.metrics import log_loss
logLOG = log_loss(y_test, y_hat_prob)


# In[ ]:





# # Model Evaluation using Test set
# 

# In[163]:


dt = {'Algorithm': ['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'],
        'Jaccard': [jacKNN, jacTREE, jacSVM, jacLOG],
     'F1-score': [f1KNN, f1TREE, f1SVM, f1LOG],
     'LOGloss': [logKNN, logTREE, logSVM, logLOG]}
df_results = pd.DataFrame(dt)
df_results


# In[ ]:





# In[ ]:




