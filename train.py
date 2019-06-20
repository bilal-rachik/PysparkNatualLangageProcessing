from pyspark.sql import SparkSession
from Pipeline import *
import  time
import warnings
import numpy as np
import mlflow
import os
import mlflow.spark

import click


@click.command(help="Trains an PYspark model on Cdiscount dataset."
                    "The input is expected in csv format."
                    "The model and its metrics are logged with mlflow.")

@click.option("--ngram", type=click.INT, default=1, help="ngram.")
@click.option("--nb_hash", type=click.INT, default=10000,help="nb_hash.")
@click.option("--maxIter", type=click.INT, default=200,help="maxIter.")

@click.option("--regParam", type=click.FLOAT, default=0.01, help="egularization L1 .")
@click.option("--elasticNetParam", type=click.FLOAT, default=0.0, help="Segularization L2.")
#@click.argument("training_data")


def train(ngram,nb_hash,maxiter,regparam,elasticnetparam):
    maxIter=maxiter
    regParam=regparam
    elasticNetParam=elasticnetparam
    warnings.filterwarnings("ignore")
    np.random.seed(40)


    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("ML") \
        .getOrCreate()
        
        
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/cdiscount_train.csv')
    
    RowDF =spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data/cdiscount_train.csv')

    # Taux de sous-échantillonnage des données pour tester le programme de préparation
    # sur un petit jeu de données
    taux_donnees=[0.80,0.19,0.01]

    dataTrain, DataTest, data_drop = RowDF.randomSplit(taux_donnees)
    n_train = dataTrain.count()
    n_test= DataTest.count()

    print("DataTrain : size = %d, DataTest : size = %d"%(n_train, n_test))
    
    
    with mlflow.start_run():
        pipeline=Pipeline_model(ngram,nb_hash,maxIter,regParam,elasticNetParam)
        

        time_start = time.time()
        # On applique toutes les étapes sur la DataFrame d'apprentissage.
        model = pipeline.fit(dataTrain)
        time_end=time.time()
        time_lrm=(time_end - time_start)
        print("LR prend %d s pour un echantillon d'apprentissage de taille : n = %d" %(time_lrm, n_train))


        predictionsDF = model.transform(DataTest)
        labelsAndPredictions = predictionsDF.select("categoryIndex","prediction").collect()
        nb_good_prediction = sum([r[0]==r[1] for r in labelsAndPredictions])
        testScore =nb_good_prediction/n_test
        print('Test score = , pour un echantillon test de taille n = %d' + str(testScore))
        
        #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        #print("score :% rmse)
    
        mlflow.log_param("ngram", ngram)
        mlflow.log_param("nb_hash",nb_hash)
        mlflow.log_param("maxIter",maxIter)
        mlflow.log_param("regParam",regParam)
        mlflow.log_param("elasticNetParam",elasticNetParam)
        mlflow.log_metric("score",testScore)
        mlflow.spark.log_model(model, "spark-model")
       

if __name__ == '__main__':
    train()