#Necesaary Imports
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC

#Import Dataset
df =pd.read_excel("HeartDisease_new.xlsx")

#Dependent and Independant Variable

X = df.drop(columns=df.columns[-1], axis =1)
y = df.iloc[:,-1]

#Train-Test-Split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

svc = SVC()

svc.fit(x_train, y_train)

svc_score = svc.score(x_test,y_test)

print(f"The initial score for SVM is {svc_score} ")

param ={"kernel":['linear', 'poly', 'rbf' ],
        'C':[.1,.4 , .6 , 1,3,5],
        'gamma':[.001,.1,.4]
    } 
grid_svm = GridSearchCV(svc , param_grid=param , verbose=3, n_jobs = -1, cv =5 )
grid_svm.fit(x_train,y_train)
optim_params = grid_svm.best_params_

print(f"The optimized parameters are : {optim_params}")

#Building a new model with the optimized parameters

optim_svc = SVC(
    kernel=optim_params['kernel'],
    C = optim_params['C'],
    gamma = optim_params['gamma'],
    )

optim_svc.fit(x_train, y_train)
final_score = optim_svc.score(x_test, y_test)

#Final score
print(f"The final score after optimizing is : {final_score}")

with open('HeartDisease_SVM_Report.txt','w') as f:
    f.write("The initial accuracy was : ")
    f.write(str(svc_score))
    f.write("\n\nThe Optimized parameters are : \n\n")
    for param, value in optim_params.items():
        f.write(param)
        f.write(" : ")
        f.write(str(value))
        f.write("\n\n")
    f.write("The final accuracy is : ")
    f.write(str(final_score))    