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
@click.option("--numTrees", type=click.INT, default=100, help="numTrees")
@click.option("--ngram", type=click.INT, default=1, help="ngram.")
@click.option("--nb_hash", type=click.INT, default=1000,help="nb_hash.")
@click.option("--word2vec", type=click.STRING, default="tf_idf",help="word2vec or tf_idf")
@click.option("--maxdepth", type=click.INT, default=16, help="numTrees")
@click.option("--Oversampling", type=click.BOOL, default=True, help="Oversampling")
def train(ngram,nb_hash,numtrees,word2vec,maxdepth,oversampling):
    numTrees =numtrees
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
    
    
    """
        df = train.fillna(0,subset='credit')
        def trasf(ligne):
            if ligne > 0:
                return 1
            else :
                  return 0
            
          
    udftrasform =  udf(lambda x: trasf(x),)
    dataClean = df.withColumn("credit_oui_non", udftrasform(df['credit']))
    dataClean.show(n=20,truncate=True)
            
    """
    df= train.select('lib1',train.credit.isNull().cast('float').alias('credit_o_n'),'lib4')

    #train.show(22,truncate=True)

    # Taux de sous-échantillonnage des données pour tester le programme de préparation
    # sur un petit jeu de données
    taux_donnees = [0.7,0.3]

    dataTrain, DataTest=df.randomSplit(taux_donnees,seed=42)
    n_train = dataTrain.count()
    n_test = DataTest.count()
    print("DataTrain : size = %d, DataTest : size = %d"%(n_train, n_test))
    if oversampling:
        def sur_echant(df, p=0.7):
            counts = df.groupBy('lib4').count().collect()
            categories = [i[0] for i in counts]
            values = [i[1] for i in counts]
            max_v = max(values)
            indx = values.index(max_v)
            dic = {j: 1 for i, j in enumerate(categories) if i != indx}
            dic[categories[indx]] = p
            df = df.sampleBy("lib4", fractions=dic)
            p = int(max_v * p)
            for i, cat in enumerate(categories):
                if i != indx:
                    a = df
                    data = a.sampleBy("lib4", fractions={cat: 1}).toPandas().sample(p - values[i], replace=True)
                    spark_df = spark.createDataFrame(data)
                    df = df.union(spark_df)
            return df
        dataTrain=sur_echant(dataTrain,p=0.8)

        print("DataTrain apres le sur_echantillonage: size = %d"%(dataTrain.count()))
    
    opm="rf"
    with mlflow.start_run():
        
        pipeline=Pipeline_model(ngram,nb_hash,data =df,opm = opm,vec=word2vec,numTrees = numTrees,maxdepth=maxdepth)

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
        #evaluator1 = MulticlassClassificationEvaluator(labelCol="categoryIndex", predictionCol="prediction", metricName="accuracy")
        #accuracy = evaluator1.evaluate(predictionsDF)
       
        evaluator = MulticlassClassificationEvaluator(labelCol="categoryIndex",predictionCol="prediction")
        f1 = evaluator.evaluate(predictionsDF,{evaluator.metricName: "f1"})
        accuracy = evaluator.evaluate(predictionsDF,{evaluator.metricName: "accuracy"})

        
        
        print("Test scor accuracy= %g" % (accuracy))
        print("Test scor f1score = %g" % (f1))
        mlflow.log_param("word2vec", word2vec)
        mlflow.log_param("ngram", ngram)
        mlflow.log_param("nb_hash", nb_hash)
        mlflow.log_param("numTrees",numTrees)
        mlflow.log_param("oversampling", oversampling)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("f1Score",f1)
        mlflow.spark.log_model(model, "spark-model")

       

if __name__ == '__main__':
    train()