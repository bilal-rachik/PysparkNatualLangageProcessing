#FROM alpine:3.1
FROM continuumio/miniconda3

#Creating an environment
ADD . /
RUN conda env create -f environment.yml

# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all

# Pull the environment name out of the environment.yml
RUN echo "source activate py36" > ~/.bashrc
ENV PATH /opt/conda/envs/py36/bin:$PATH

# JAVA
RUN apt-get update \
 && apt-get install -y openjdk-8-jre \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# HADOOP
ENV HADOOP_VERSION 2.7.1
ENV HADOOP_HOME /usr/hadoop-$HADOOP_VERSION
ENV PATH $PATH:$HADOOP_HOME/bin

RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz \
  && tar -xvf hadoop-2.7.3.tar.gz \
  && mv hadoop${HADOOP_VERSION} spark \
  && rm hadoop${HADOOP_VERSION}.tgz
  && cd /

#SPARK
ENV SPARK_VERSION 2.4.3
ENV SPARK_HOME /spark/spark-${SPARK_VERSION}
ENV PATH $PATH:${SPARK_HOME}/bin
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && cd /


ENV PYTHONPATH ${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.7-src.zip


# Bundle app source
EXPOSE  80
CMD ["python", "prediction.py"]
