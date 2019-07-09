from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import nltk
from pyspark.ml.feature import StopWordsRemover
class MyNltkStemmer(Transformer, HasInputCol, HasOutputCol):
   
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(MyNltkStemmer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        STEMMER = nltk.stem.SnowballStemmer('french')
        def clean_text(tokens):
            tokens_stem = [ STEMMER.stem(token) for token in tokens]
            return tokens_stem
        udfCleanText =  udf(lambda lt : clean_text(lt), ArrayType(StringType()))
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udfCleanText(in_col))
    

def remover(inputCol,outputCol):
    s=set(nltk.corpus.stopwords.words('french'))
    lucene_stopwords =open("data/lucene_stopwords.txt","r").read().split(",") #En local
    ## Union des deux fichiers de stopwords 
    stopwords = list(s.union(set(lucene_stopwords)))
    return StopWordsRemover(inputCol=inputCol, outputCol=outputCol, stopWords =stopwords)





