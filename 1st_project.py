import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('data.csv')
'''print(df.shape)
print(df.head())
print(df.info())#to tell datatypes used
print(df.describe())
print(df.isnull().sum())#no null values are present
numeric_column=['age','bmi','children','charges']
for cols in numeric_column:
    plt.figure(figsize=(6,4))
    sns.histplot(df[cols],kde=True,bins=20)#kde=kernel density estimation(to see skewness)
    plt.show()
sns.countplot(x=df['children']) #countplot for categorical or non numeric data 
plt.show()  
sns.countplot(x=df['sex'])
plt.show()
sns.countplot(x=df['smoker'])
plt.show()
for cols in numeric_column:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[cols])#boxplots to see outliers.
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()'''

#DATA CLEANING AND PREPROCESSING.
df_cleaned=df.copy()#to create copy of original data.
df_cleaned.drop_duplicates(inplace=True)#to drop any duplicate rows.
#print(df_cleaned.shape)#in output we can see one row was dropped.
#print(df_cleaned['sex'].value_counts())#to fix any error in category
#print(df_cleaned['smoker'].value_counts())
#print(df_cleaned['region'].value_counts())
#so output of above three showed that there is no error in category lower or uppercasing
#DATA PREPROCESSING.
#Label Encoding.
df_cleaned['sex']=df_cleaned['sex'].map({"female":1,"male":0})
df_cleaned['smoker']=df_cleaned['smoker'].map({"yes":1,"no":0})
#One Hot Encoding for regions-we will create different columns for diff region.
df_cleaned=pd.get_dummies(df_cleaned,columns=['region'])
df_cleaned=df_cleaned.astype(int)#to convert boolean of region to 0s and 1s
#Feature engineering->
#create new columns and test efficiency,remove default columns and then test efiiciency
#we will create new columns for bmi acc to bmi classes.
df_cleaned['bmi_category']=pd.cut(#cut function is used to categorize
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
) 
df_cleaned=pd.get_dummies(df_cleaned,columns=['bmi_category'])
df_cleaned=df_cleaned.astype(int)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler=StandardScaler()
# Scale numerical columns
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
final_df=df_cleaned[['age','sex','bmi','children','smoker','charges','region_southwest','bmi_category_Obese']]#By chi square test we isolated them.
#Now we can apply linear regression algo for prediction,but before that we have to divide our data
#in 80 percent training and 20 percent for testing 
from sklearn.model_selection import train_test_split
x=final_df.drop('charges',axis=1)#to isolate input features.
y=final_df['charges']
x_train, x_test, y_train, y_test = train_test_split(   x, y, test_size=0.20, random_state=42)#we have splitted 
#x and y in train and test ,we will make predictions with x_test and stotre it in y_pred and compare it with y_test.
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#print(y_pred)
#print(y_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)#to test accuracy of our model
n=x_test.shape[0]
p=x_test.shape[1]
r2_adjusted=1-((1-r2)*(n-1)/(n-p-1))
print(r2_adjusted)
print(r2)
#Now we can deploy our model as our accuracy is great
import joblib

joblib.dump(model, "charges_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(mae)














