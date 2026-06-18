import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('heart.csv')
#print(df.head())
#print(df.info())
numeric_cols=['Age','RestingBP','Cholesterol','MaxHR']
'''for cols in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[cols],kde=True)
    plt.show()#A person's cholestrol cannot be zero so we will replace it by its mean'''
ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol']=df['Cholesterol'].round(2)
rb_mean=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0,rb_mean)
df['RestingBP']=df['RestingBP'].round(2)
'''sns.countplot(x=df['ChestPainType'],hue=df['HeartDisease'])
plt.show()
sns.countplot(x=df['Sex'],hue=df['HeartDisease'])
plt.show()
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()'''
df_encoded=pd.get_dummies(df)
df_encoded=df_encoded.astype(int)
print(df_encoded.head())
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
df_encoded[cols]=scaler.fit_transform(df_encoded[cols])
print(df_encoded.head())

#After checking F1 score of every algo KNN suits best for this project
x=df_encoded.drop(columns=['HeartDisease'])
y=df_encoded['HeartDisease']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))

import joblib

joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(x.columns.tolist(), "heart_feature_columns.pkl")

print("Files saved successfully!")









