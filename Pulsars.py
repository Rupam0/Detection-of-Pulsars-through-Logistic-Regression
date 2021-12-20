#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


data=pd.read_csv(r"C:\Users\Rupam\Desktop\pulsar_stars.csv")


# In[28]:


data


# In[4]:


data.info()


# In[5]:


data = data.rename(columns={' Mean of the integrated profile':"mean_integrated_profile",
       ' Standard deviation of the integrated profile':"std_deviation_integrated_profile",
       ' Excess kurtosis of the integrated profile':"kurtosis_integrated_profile",
       ' Skewness of the integrated profile':"skewness_integrated_profile", 
        ' Mean of the DM-SNR curve':"mean_dm_snr_curve",
       ' Standard deviation of the DM-SNR curve':"std_deviation_dm_snr_curve",
       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dm_snr_curve",
       ' Skewness of the DM-SNR curve':"skewness_dm_snr_curve",
       })


# In[33]:


data.head()


# In[30]:


f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linecolor="blue",fmt=".2f",ax=ax)
plt.show()


# In[13]:


g = sns.pairplot(data, hue="target_class",palette="husl",diag_kind = "kde",kind = "scatter")


# In[14]:


y = data["target_class"].values
x_data = data.drop(["target_class"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


# Logistic Regression

# In[16]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_prediction = lr.predict(x_test)


# In[17]:


from sklearn.metrics import mean_squared_error
mse_lr=mean_squared_error(y_test,lr_prediction)

from sklearn.metrics import confusion_matrix,classification_report
cm_lr=confusion_matrix(y_test,lr_prediction)
cm_lr=pd.DataFrame(cm_lr)
cm_lr["total"]=cm_lr[0]+cm_lr[1]
cr_lr=classification_report(y_test,lr_prediction)


# In[18]:


from sklearn.metrics import cohen_kappa_score
cks_lr= cohen_kappa_score(y_test, lr_prediction)


# In[19]:


score_and_mse={"model":["logistic regression"],"Score":[lr.score(x_test,y_test)],"Cohen Kappa Score":[cks_lr],"MSE":[mse_lr]}
score_and_mse=pd.DataFrame(score_and_mse)


# In[20]:


print('Classification report for Logistic Regression:',cr_lr)


# In[25]:


f, axes = plt.subplots(2, 3,figsize=(18,12))
g1 = sns.heatmap(cm_lr,annot=True,fmt=".1f",cmap="flag",cbar=False,ax=axes[0,0])
g1.set_ylabel('y_true')
g1.set_xlabel('y_head')
g1.set_title("Logistic Regression")


# In[26]:


score_and_mse


# In[ ]:




