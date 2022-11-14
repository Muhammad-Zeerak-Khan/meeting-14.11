#Necessary Imports
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


#Data Import
df = pd.read_excel("HeartDisease_new.xlsx") 

#Dependent and Independant Variable

X = df.drop(columns=df.columns[-1], axis =1)
y = df.iloc[:,-1]

#Train-Test-Split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

rf = RandomForestClassifier(n_estimators=5)

rf.fit(x_train, y_train)
rf_score = rf.score(x_test,y_test)

print(f"The initial score for Random Forest is {rf_score} ")

grid_pram = {
    "n_estimators" : [5,10 , 50 , 100 , 120 , 150],
    'criterion' :['gini' ,'entropy'],
    'max_depth' :range(2,10),
    "min_samples_split":range(2,10 ,1),
    'min_samples_leaf' :range(2,10)
    
}

grid_rf = GridSearchCV(param_grid= grid_pram, cv = 5 , n_jobs=-1,verbose=3 ,estimator = rf)

grid_rf.fit(x_train, y_train)


optim_params = grid_rf.best_params_

print(f"The optimized parameters are : {optim_params}")

#Building a new model with the optimized parameters

optim_rf = RandomForestClassifier(
    criterion=optim_params['criterion'],
    max_depth = optim_params['max_depth'],
    min_samples_leaf = optim_params['min_samples_leaf'],
    min_samples_split = optim_params['min_samples_split']
    )

optim_rf.fit(x_train, y_train)
final_score = optim_rf.score(x_test, y_test)

#Final score
print(f"The final score after optimizing is : {final_score}")

with open('HeartDisease_RandomForest_Report.txt','w') as f:
    f.write("The initial accuracy was : ")
    f.write(str(rf_score))
    f.write("\n\nThe Optimized parameters are : \n\n")
    for param, value in optim_params.items():
        f.write(param)
        f.write(" : ")
        f.write(str(value))
        f.write("\n\n")
    f.write("The final accuracy is : ")
    f.write(str(final_score))