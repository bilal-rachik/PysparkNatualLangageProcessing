#FROM alpine:3.1
FROM continuumio/miniconda3

ADD . /

# JAVA
RUN tar -xvf jdk-8u211-linux-x64.tar.gz \
   && mv jdk-8u211-linux-x64 spark \
   && rm jdk-8u211-linux-x64.tar.gz \
   && cd /

ENV JAVA_HOME /spark/jdk-8u211-linux-x64/jdk1.8.0_211
ENV PATH $PATH:$JAVA_HOME/bin

#SPARK
ENV SPARK_VERSION 2.4.3
ENV SPARK_HOME /spark/spark-${SPARK_VERSION}
ENV PATH $PATH:${SPARK_HOME}/bin

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz \
    && tar -xvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz \
    && cd /

# HADOOP
ENV HADOOP_VERSION 2.7.1
ENV HADOOP_HOME /spark/hadoop-$HADOOP_VERSION
ENV PATH $PATH:$HADOOP_HOME/bin

RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz \
  && tar -xvf hadoop-${HADOOP_VERSION}.tar.gz \
  && mv hadoop-${HADOOP_VERSION} spark \
  && rm hadoop-${HADOOP_VERSION}.tar.gz \
  && cd /

#Creating an environment

RUN conda env create -f environment.yml

# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all

# Pull the environment name out of the environment.yml
RUN echo "source activate py36" > ~/.bashrc
ENV PATH /opt/conda/envs/py36/bin:$PATH


ENV PYTHONPATH ${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.7-src.zip

# Bundle app source
EXPOSE  80
CMD ["python", "prediction.py"]
