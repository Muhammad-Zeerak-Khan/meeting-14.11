import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#Data Import
df = pd.read_excel("HeartDisease_new.xlsx") 

#Dependent and Independant Variable
X = df.drop('HeartDisease', axis =1)
y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state= 42)
knn = KNeighborsClassifier()

knn.fit(x_train,y_train)
knn_score = knn.score(x_test, y_test)
print(f"The initial score for KNN is {knn_score} ")

grid_param  = {
    'n_neighbors':[3,5,7,9,12,13,15,17,21],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size' : [10 , 15 , 20 , 25 , 30 , 35 , 45 , 50 ],
    'p' : [1,2],
    'weights' : ['uniform', 'distance']
    
    
    
}
grid_knn = GridSearchCV(knn,param_grid=grid_param, cv =5, n_jobs=-1 )

grid_knn.fit(x_train, y_train)

optim_params = grid_knn.best_params_


print(f"The optimized parameters are : {optim_params}")

#Building a new model with the optimized parameters
optim_knn = KNeighborsClassifier(
    n_neighbors=optim_params["n_neighbors"],
    algorithm = optim_params["algorithm"],
    leaf_size = optim_params["leaf_size"],
    p = optim_params["p"],
    weights = optim_params["weights"]
    )

optim_knn.fit(x_train,y_train)

final_score = optim_knn.score(x_test,y_test)

with open('HeartDisease_Knn_Report.txt','w') as f:
    f.write("The initial accuracy was : ")
    f.write(str(knn_score))
    f.write("\n\nThe Optimized parameters are : \n\n")
    for param, value in optim_params.items():
        f.write(param)
        f.write(" : ")
        f.write(str(value))
        f.write("\n\n")
    f.write("The final accuracy is : ")
    f.write(str(final_score))    
