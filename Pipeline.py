from Outils import remover
from Outils import MyNltkStemmer
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
##Regression logistique
from pyspark.ml.classification import LogisticRegression
##Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier
##Random Forest
from pyspark.ml.classification import RandomForestClassifier 
##Pour la création des DataFrames
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import NGram

def Pipeline_model(ngram,nb_hash,maxIter0,regParam0,elasticNetParam0):
    # Fonction tokenizer qui permet de remplacer un long texte par une liste de mot
    regexTokenizer = RegexTokenizer(inputCol="Description", outputCol="tokenizedDescr", pattern="[^a-z_]",minTokenLength=3, gaps=True)
    # Fonction StopWordsRemover qui permet de supprimer des mots
    remover1=remover(inputCol="tokenizedDescr",outputCol="stopTokenizedDescr")
    # Stemmer 
    #stemmer = MyNltkStemmer(inputCol="stopTokenizedDescr", outputCol="cleanDescr")
     # Define NGram transformer
    ngram= NGram(n=ngram, inputCol="stopTokenizedDescr", outputCol="bigrams")
    # Indexer
    indexer = StringIndexer(inputCol="Categorie1", outputCol="categoryIndex")
    # Hasing
    hashing_tf = HashingTF(inputCol="bigrams", outputCol='tf', numFeatures=nb_hash)
    # Inverse Document Frequency
    idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="tfidf")

    #Logistic Regression
    lr = LogisticRegression(maxIter=maxIter0,regParam=regParam0, fitIntercept=False, tol=0.0001,
            family = "multinomial", elasticNetParam=elasticNetParam0, featuresCol="tfidf", labelCol="categoryIndex") #0 for L2 penalty, 1 for L1 penalty

    # Creation du pipeline
    pipeline = Pipeline(stages=[regexTokenizer, remover1,ngram,indexer, hashing_tf, idf, lr ])
    return pipeline 























