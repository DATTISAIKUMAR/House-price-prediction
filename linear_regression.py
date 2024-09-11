import pandas as pd
df=pd.read_csv("data.csv")
print(df.isnull().sum())
df=df.drop(["date","street","statezip","country"],axis=1)
print(df.dtypes)
from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
df["city"]=le.fit_transform(df["city"])
import matplotlib.pyplot as plt
import seaborn as sns

for i in df.columns:
    sns.boxplot(df[i])
    plt.title(f"{i}")
    plt.show()

def remove(df,i):

    q25=df[i].quantile(0.25)
    q75=df[i].quantile(0.75)
    iqr=q75-q25
    lower=q25-1.5*iqr
    upper=q75+1.5*iqr
    df=df[df[i]>=lower]
    df=df[df[i]<=upper]
    return df

for i in df.columns:
    df=remove(df,i)


import seaborn as sns

for i in df.columns:
    sns.boxplot(df[i])
    plt.title(f"{i}")
    plt.show()

y=df.iloc[:,0]
x=df.iloc[:,1:]


sc=StandardScaler()
x=sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1000)



from sklearn.neighbors import KNeighborsRegressor
knr=KNeighborsRegressor()
knr.fit(x_train,y_train)
from sklearn.tree import DecisionTreeRegressor
DT=DecisionTreeRegressor()
DT.fit(x_train,y_train)

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)

from sklearn.svm import SVR
SVR=SVR()
SVR.fit(x_train,y_train)


import joblib
joblib.dump(knr,"KNN")
joblib.dump(DT,"decision tree")
joblib.dump(LR,"Linear Regression")
joblib.dump(SVR,"SVM")
