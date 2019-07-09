from pyspark.sql import SparkSession
import mlflow.spark
import pandas as pd
import mlflow.pyfunc
from flask import Flask ,jsonify,request,json

app = Flask(__name__)
spark = SparkSession.builder \
        .master("local[*]") \
        .appName("ML") \
        .getOrCreate()
        
import numpy as np
def jsntodf(jsn):
    df = pd.DataFrame.from_dict(jsn, orient='columns')
    df.debit[df.debit==""]=np.NaN
    df.credit[df.credit==""]=np.NaN
    df['debit'] = df.debit.astype(float)
    df['credit'] = df.credit.astype(float)
    df['credit_o_n']=df.credit.isnull()
    df['credit_o_n'] = df.credit_o_n.astype(float)
    return df
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/categorize', methods=['POST'])
def pred():
    df=jsntodf(request.json)
    df=spark.createDataFrame(df)
    model_path_dir ="mlruns/0/c1a2ff5b99c246f19ee6e2df4d1406fe/artifacts/spark-model"
    model=mlflow.spark.load_model(model_path_dir)
    prd=model.transform(df)
    dfp=prd.select("lib1","credit","debit","predictedLabel")
    df=dfp.toPandas()
    df=df.to_json(orient='records')
    #result={}
    #for index, row in df.iterrows():
    #    result[index] = row.to_json()
    #   result[index] = dict(row)
    
    return json.jsonify(df)
if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)
























    