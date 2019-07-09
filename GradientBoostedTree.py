from pyspark.sql import SparkSession
from Pipeline import *
import  time
import warnings
import numpy as np
import mlflow
import os
import mlflow.spark
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import click


@click.command(help="Trains an PYspark model on Cdiscount dataset."
                    "The input is expected in csv format."
                    "The model and its metrics are logged with mlflow.")
@click.option("--ngram", type=click.INT, default=1, help="ngram.")
@click.option("--nb_hash", type=click.INT, default=1000,help="nb_hash.")
@click.option("--maxIter", type=click.INT, default=20,help="maxIter.")
@click.option("--word2vec", type=click.STRING, default="tf_idf",help="word2vec or tf_idf")
#@click.argument("training_data")


def train(ngram,nb_hash,maxiter,word2vec):
    maxIter = maxiter
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("ML") \
        .getOrCreate()
        
        
    #wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/cdiscount_train.csv')
    RowDF = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true',sep=",").load('data/operation bancaire.csv')
    train = RowDF.dropna(subset='lib4')
    print("nbr de classe",train.select(train.columns[4]).distinct().count())

    df= train.select('lib1',train.credit.isNull().cast('float').alias('credit_o_n'),'lib4')
    
    # Taux de sous-échantillonnage des données pour tester le programme de préparation
    # sur un petit jeu de données
    taux_donnees=[0.7,0.3]

    dataTrain, DataTest= df.randomSplit(taux_donnees)
    n_train = dataTrain.count()
    n_test = DataTest.count()

    print("DataTrain : size = %d, DataTest : size = %d"%(n_train, n_test))
    
    opm="gpt"
    
    with mlflow.start_run():
       
        pipeline=Pipeline_model(ngram,nb_hash,data=df,opm=opm,vec=word2vec,maxIter=maxIter)
        

        time_start = time.time()
        # On applique toutes les étapes sur la DataFrame d'apprentissage.
        model = pipeline.fit(dataTrain)
        time_end = time.time()
        time_lrm = (time_end - time_start)
        print("LR prend %d s pour un echantillon d'apprentissage de taille : n = %d" %(time_lrm, n_train))

        predictionsDF = model.transform(DataTest)

        #labelsAndPredictions = predictionsDF.select("categoryIndex","prediction").collect()
        #nb_good_prediction = sum([r[0]==r[1] for r in labelsAndPredictions])
        #testScore =nb_good_prediction/n_test
        #print('Test score = , pour un echantillon test de taille n = %d' + str(testScore))
        
         # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(labelCol="categoryIndex", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictionsDF)
        print("Test scor = %g" % (accuracy))
          
        mlflow.log_param("word2vec",word2vec)
        mlflow.log_param("ngram", ngram)
        mlflow.log_param("nb_hash",nb_hash)
        mlflow.log_param("maxIter",maxIter)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.spark.log_model(model, "spark-model")
       

if __name__ == '__main__':
    train()
