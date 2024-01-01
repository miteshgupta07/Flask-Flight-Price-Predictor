import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle

data=pd.read_csv("E:\Data Science\ML Deployment\Heroku Deployment\deploy_data.csv")

x=data.drop('Price',axis=1)
y=data['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

model=CatBoostRegressor()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(y_pred)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))