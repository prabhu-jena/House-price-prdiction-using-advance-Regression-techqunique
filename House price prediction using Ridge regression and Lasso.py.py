#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[7]:


df_1=pd.read_csv("D:\\data\\house price data\\train.csv")
df_2=pd.read_csv("D:\\data\\house price data\\test.csv")


# In[8]:


df_1.head()


# In[9]:


df_2.head


# In[12]:


# A exploratory view on correlation among variables(between columns of train data)


# In[103]:


corcf=df_1.corr()
corcf


# In[73]:


#Exploring the column saleprice which is nothing but a target variable
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# In[78]:


sns.distplot(df_1["SalePrice"],fit=norm);
fig=plt.figure()
res=stats.probplot(df_1["SalePrice"],plot=plt)
plt.show
             
             


# In[79]:


#The saleprice is not normally distributed so we use logarthimic transformation for more accurate curve


# In[80]:


df_1["SalePrice"]=np.log(df_1["SalePrice"])


# In[82]:


sns.distplot(df_1["SalePrice"],fit=norm);
fig=plt.figure()
res=stats.probplot(df_1["SalePrice"],plot=plt)
plt.show


# In[83]:


# Exploring columns by correlation among themselves....(the high correlation against saleprice taken as first priority)


# In[85]:


# The main importance features are 
      #OverallQual.
       #YearBuilt.
       #TotalBsmtSF.
       #GrLivArea.
       #GarageCars.


# In[95]:


sns.set
cols=["SalePrice","OverallQual","YearBuilt","TotalBsmtSF","GrLivArea","GarageCars"]
sns.pairplot(df_1[cols],size=2.5)
plt.show();


# In[97]:


# in order to a high relation between GrLivArea and saleprice ,we have to explore these variables


# In[100]:


sns.pairplot(df_1[["SalePrice","GrLivArea"]])


# In[101]:


# To precise the data, we have to remove the outliers 


# In[107]:


df_1.drop(df_1[(df_1["SalePrice"]<3000)& (df_1["GrLivArea"]>4000)].index,inplace=True)


# In[114]:


colls=corcf.nlargest(10,"SalePrice")["SalePrice"].index
clm=np.corrcoef(df_1[colls].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(clm, cbar=True, annot=True, 
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=colls.values, xticklabels=colls.values)
plt.show()


# In[120]:


target=df_1.SalePrice.copy()


# In[121]:


#Deal with missing data


# In[124]:


df=pd.concat((df_1,df_2)).reset_index(drop=True)
df.drop(["SalePrice"],axis=1,inplace=True)
print("all_data size is : {}".format(df.shape))


# In[125]:


df.info()


# In[126]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[130]:


#Inputting missing values


# In[131]:


for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"] :
    df[col] = df[col].fillna("No")


# In[132]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna('No')


# In[133]:


for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    df[col] = df[col].fillna(0)


# In[134]:


for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    df[col] = df[col].fillna(0)


# In[135]:


for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    df[col] = df[col].fillna('No')


# In[136]:


df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)


# In[137]:


df.MSZoning.hist()


# In[138]:


#I'm going to fill the MSZoning with RL the most frequent value.


# In[140]:


df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df["Functional"] = df["Functional"].fillna("Typ")


# In[141]:


#the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , i can fill the missing values by the median LotFrontage of the neighborhood.


# In[142]:


df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[144]:


#Again checking the missing value in df


# In[143]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[146]:


df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['MSSubClass'] = df['MSSubClass'].fillna("No")
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])


# In[147]:


df.drop(["Utilities"], axis=1, inplace=True)


# In[149]:


#changing the data types & encoding


# In[150]:


for col in ["MSSubClass", "OverallCond", "YrSold", "MoSold"]:
    df[col] = df[col].apply(str)


# In[152]:


# give a ordinal encoding for these variables


# In[153]:


from sklearn.preprocessing import LabelEncoder


# In[154]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))


# In[155]:


#check skewness


# In[157]:


from scipy.stats import norm, skew


# In[158]:


numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness.head(10)


# In[159]:


#applying Box-cox transformation


# In[160]:


from scipy.special import boxcox1p


# In[161]:


skewed_features = skewness.index
lam = 0.15
for col in skewed_features:
    df[col] = boxcox1p(df[col], lam)


# In[162]:


#Applying a hot encoding


# In[163]:


df.drop(["Id"], axis=1, inplace=True)


# In[164]:


df = pd.get_dummies(df)


# In[165]:


train = df[:df_1.shape[0]]
test = df[df_1.shape[0]:]


# In[166]:


num_vars = train.select_dtypes(include=['int64','float64']).columns


# In[167]:


scaler = StandardScaler()
train[num_vars] = scaler.fit_transform(train[num_vars])
test[num_vars] = scaler.transform(test[num_vars])


# In[168]:


#Applying training models


# In[171]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[172]:


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)


# In[173]:


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
                 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100]
         }


ridge = Ridge()

folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_squared_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)    

model_cv.fit(X_train, y_train) 


# In[174]:


ridge = model_cv.best_estimator_


# In[175]:


y_train_pred_ridge = ridge.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred_ridge))


# In[176]:


y_test_pred_ridge = ridge.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred_ridge))


# In[177]:


print ('RMSE Validation is: \n', mean_squared_error(y_test, y_test_pred_ridge))


# In[179]:


#Trying Lasso Regression


# In[180]:


params = {'alpha': [0.00005, 0.0001, 0.001, 0.008, 0.01]}
lasso = Lasso()

model_cv_l = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_squared_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv_l.fit(X_train, y_train)


# In[181]:


lasso = model_cv_l.best_estimator_


# In[182]:


y_train_pred_lasso = lasso.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred_lasso))


# In[183]:


y_test_pred_lasso = lasso.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred_lasso))


# In[184]:


print ('RMSE Validation is: \n', mean_squared_error(y_test, y_test_pred_lasso))


# In[185]:


#preparing results


# In[186]:


preds = np.exp(ridge.predict(test))


# In[188]:


predictions = pd.DataFrame({'Id': df_2['Id'] ,'SalePrice': preds })


# In[189]:


predictions.to_csv("preds.csv",index=False)


# In[190]:


predictions.SalePrice


# In[ ]:





# In[ ]:




