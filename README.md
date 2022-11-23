# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: P Hariharan
RegisterNumber: 212220040038
``` 
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Data Head:
![head](https://user-images.githubusercontent.com/94828604/173539464-d9fb4ed7-afe5-40e0-8c6e-5f1e9601fcbd.png)


Information:
![info](https://user-images.githubusercontent.com/94828604/173539565-023822db-cbaf-4c2a-a113-16740b26bfd8.png)



Null dataset:
![null](https://user-images.githubusercontent.com/94828604/173539625-9de935de-93a7-4187-8dc7-2c80af2f561f.png)


Value_counts():
![left](https://user-images.githubusercontent.com/94828604/173539666-4b82e0e6-f538-4b5e-a650-83e3169dae9a.png)


Data Head:
![head2](https://user-images.githubusercontent.com/94828604/173539713-1293a3fb-661f-4651-8aa0-1c6f23264259.png)

x.head():
![xhead](https://user-images.githubusercontent.com/94828604/173539761-59bde7f3-88e9-42c0-a676-39dbc6db5308.png)


Accuracy:
![accuracy](https://user-images.githubusercontent.com/94828604/173539815-b1fb091a-2aa3-4062-ae8b-67c143adad4d.png)


Data Prediction:
![predict](https://user-images.githubusercontent.com/94828604/173539886-b8b8845e-bfeb-49d9-9ab1-4f61735a8c54.png)



## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.


