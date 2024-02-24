# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle

# Loading Dataset
data=pd.read_csv("deploy_data.csv")

# Splitting the Dataset
x=data.drop('Price',axis=1)
y=data['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

# Model Building
model=CatBoostRegressor()

# Training the Model
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(y_pred)

# Saving the Model
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
