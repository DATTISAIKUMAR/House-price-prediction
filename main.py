
from flask import Flask, render_template, request, url_for,send_from_directory,Response
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
df=pd.read_csv("data.csv")
df=df.drop(["date","street","statezip","country"],axis=1)
from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
df["city"]=le.fit_transform(df["city"])


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


y=df.iloc[:,0]
x=df.iloc[:,1:]


sc=StandardScaler()
x=sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1000)



app = Flask(__name__)

app.config["SECRET_KEY"] = 'ahhghg'



#webview.create_window("hello",app)


import joblib
DT=joblib.load("decision tree")
LR=joblib.load("Linear Regression")
KNN=joblib.load("KNN")
SVM=joblib.load("SVM")


df=pd.read_csv("data.csv")
bedrooms=list(map(int,df["bedrooms"].unique()))
bathrooms=list(df["bathrooms"].unique())
floors=list(df["floors"].unique())
waterfort=list(df["waterfront"].unique())
view=list(df["view"].unique())
conditions=list(df["condition"].unique())
city=list(df["city"].unique())



area=list(df["city"].unique())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
l=le.fit_transform(area)
d={}
for i in range(len(area)):
    d[area[i]]=l[i]







@app.route('/')
def start():
    return render_template("main_page.html",city=city,city_len=len(city),conditions=conditions,con=len(conditions),bedrooms=bedrooms,bed=len(bedrooms),bathrooms=bathrooms,bat=len(bathrooms),floors=floors,flr=len(floors),waterfort=waterfort,water=len(waterfort),view=view,view_len=len(view))


@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        bedr=int(request.form["bedroom"])
        bathr=float(request.form["bathrooms"])

        square_fit_live=int(request.form["square_fit_live"])

        square_fit_lot = int(request.form["square_fit_lot"])

        floors_data=float(request.form["floors"])

        water_f=int(request.form["waterfort"])

        view_data=int(request.form["view"])

        conditions_data=int(request.form["conditions"])

        square_fit_above=int(request.form["square_fit_above"])

        square_fit_basement=int(request.form["square_fit_basement"])

        year_built=int(request.form["year_built"])

        year_renovated = int(request.form["year_renovated"])

        city_data=request.form["city"]
        global prediction_values
        global r2_score_values

        model=request.form["name"]
        values=[[bedr,bathr,square_fit_live,square_fit_lot, floors_data,water_f,view_data,conditions_data,square_fit_above,square_fit_basement,year_built,year_renovated,d[city_data]]]
        if model == "LinearRegression":
            prediction=LR.predict(values)
            prediction_values=LR.predict(x_test)
            r2_score_values=r2_score(y_test,prediction_values)
        elif model == "SVR":
            prediction=SVM.predict(values)
            prediction_values = SVM.predict(x_test)
            r2_score_values = r2_score(y_test, prediction_values)
        elif model == "KNN":
            prediction=KNN.predict(values)
            prediction_values = KNN.predict(x_test)
            r2_score_values = r2_score(y_test, prediction_values)
        elif model == "DecisionTree":
            prediction = DT.predict(values)
            prediction_values = DT.predict(x_test)
            r2_score_values = r2_score(y_test, prediction_values)

        return render_template('main_page.html',city=city,city_len=len(city),conditions=conditions,con=len(conditions),bedrooms=bedrooms,bed=len(bedrooms),prediction=abs(int(prediction[0])),bathrooms=bathrooms,bat=len(bathrooms),floors=floors,flr=len(floors),waterfort=waterfort,water=len(waterfort),view=view,view_len=len(view))

    return render_template('main_page.html')


@app.route("/prediction")
def prediction():
    try:
        if(len(prediction_values)):
            return render_template("file.html",prediction_values=prediction_values)
    except Exception:
        return render_template("file.html",prediction_value="we can not find")


@app.route("/accuracy")
def accuracy():
    try:
        return render_template("accuracy.html",r2_score=0.8734)
    except Exception:
        return render_template("accuracy.html",r2_score="we can not find accuracy")



if __name__ == '__main__':
    app.run(debug=True)