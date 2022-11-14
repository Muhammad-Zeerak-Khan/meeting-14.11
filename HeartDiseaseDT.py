import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#Data Import
df = pd.read_excel("diabetes.xlsx") 

#Dependent and Independant Variable

X = df.drop(columns=df.columns[-1], axis =1)
y = df.iloc[:,-1]

#Train-Test-Split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

#Decison Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

dt_score = dt_model.score(x_test,y_test)
print(f"The initial score for Decision Tree is {dt_score} ")

# #Hyper parameter optimization

# grid_pram = {"criterion":['gini','entropy'],
#               "splitter":['best','random'],
#               "max_depth" : range(2,40,1),
#               "min_samples_split":range(2,10 ,1),
#               "min_samples_leaf":range(1,10,1),
#               'ccp_alpha':np.random.rand(20)
#               }


# #Finding the optimized parameters
# grid_dt = GridSearchCV(estimator = dt_model, param_grid = grid_pram,
#                                   cv = 5, n_jobs = -1 , verbose = 3)

# grid_dt.fit(x_train, y_train)
# optim_params = grid_dt.best_params_

# print(f"The optimized parameters are : {optim_params}")

# #Building a new model with the optimized parameters

# optim_dt = DecisionTreeClassifier(
#     criterion=optim_params['criterion'],
#     max_depth = optim_params['max_depth'],
#     min_samples_leaf = optim_params['min_samples_leaf'],
#     min_samples_split = optim_params['min_samples_split'],
#     splitter = optim_params['splitter'],
#     )

# optim_dt.fit(x_train, y_train)
# final_score = optim_dt.score(x_test, y_test)

# #Final score
# print(f"The final score after optimizing is : {final_score}")

# with open('HeartDisease_DecisionTree_Report.txt','w') as f:
#     f.write("The initial accuracy was : ")
#     f.write(str(dt_score))
#     f.write("\n\nThe Optimiezed parameters are : \n\n")
#     for param, value in optim_params.items():
#         f.write(param)
#         f.write(" : ")
#         f.write(str(value))
#         f.write("\n\n")
#     f.write("The final accuracy is : ")
#     f.write(str(final_score))    