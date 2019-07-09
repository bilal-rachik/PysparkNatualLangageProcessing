from Outils import remover
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
##Regression logistique
from pyspark.ml.classification import LogisticRegression
##Random Forest
from pyspark.ml.classification import RandomForestClassifier
#dradient posting 
from pyspark.ml.classification import GBTClassifier
##Pour la création des DataFrames
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StringIndexer,IndexToString
from pyspark.ml.feature import NGram
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import MultilayerPerceptronClassifier

def Pipeline_model(ngram,nb_hash,data,opm,vec="tf_idf",maxIter=200,regParam =0.01,elasticNetParam=0.0,numTrees = 200,maxdepth=16):
    # Fonction tokenizer qui permet de remplacer un long texte par une liste de mot
    regexTokenizer = RegexTokenizer(inputCol="lib1", outputCol="tokenizedDescr", pattern="[^a-z_]",minTokenLength=3, gaps=True)
    
    # Fonction StopWordsRemover qui permet de supprimer des mots
    remover1 = remover(inputCol="tokenizedDescr",outputCol="stopTokenizedDescr")
    # Stemmer
    #stemmer = MyNltkStemmer(inputCol="stopTokenizedDescr", outputCol="cleanDescr")
     # Define NGram transformer
    ngram1= NGram(n=ngram, inputCol="stopTokenizedDescr", outputCol="bigrams")
    # Indexer
    indexer = StringIndexer(inputCol="lib4", outputCol="categoryIndex").fit(data)
    if vec == "tf_idf":
        # Hasing
        hashing_tf = HashingTF(inputCol="bigrams", outputCol='tf', numFeatures=nb_hash)
        # Inverse Document Frequency
        idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="tfidf")
    
    else:
        Word2Vec_ = Word2Vec(vectorSize=300,minCount=0,inputCol="bigrams", outputCol="tfidf",seed=42)
    
    assembler = VectorAssembler(inputCols=["tfidf","credit_o_n"],outputCol="features")

    #Logistic Regression
    if opm == "rl" :
        print("use LogisticRegression")
        model = LogisticRegression(maxIter = maxIter,regParam = regParam, fitIntercept=False, tol = 0.0001,
            family = "multinomial", elasticNetParam=elasticNetParam, featuresCol="features", labelCol="categoryIndex")
    elif opm == "rf" :
        print("RandomForestClassifier")
        model = RandomForestClassifier(labelCol="categoryIndex", featuresCol="features", numTrees=numTrees,
                                       maxMemoryInMB=1000,maxDepth=maxdepth,seed=42)
    elif opm=="mp":
        print("MultilayerPerceptronClassifier")
        model= MultilayerPerceptronClassifier(maxIter=maxIter,labelCol="categoryIndex", featuresCol="features", layers=[1001,70,26],blockSize=128, seed=42)

    else :
        print("GradientBoostedTree")
        model = GBTClassifier(labelCol="categoryIndex", featuresCol="features",maxIter=maxIter,seed=42)
    # Convertion  indexed labels en originale labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=indexer.labels)
    # Creation du pipeline
    if vec == "tf_idf":
        return Pipeline(stages=[regexTokenizer, remover1,ngram1,indexer, hashing_tf, idf, assembler,model,labelConverter])
    else :
        return Pipeline(stages=[regexTokenizer, remover1,ngram1,indexer,Word2Vec_,assembler,model,labelConverter])























