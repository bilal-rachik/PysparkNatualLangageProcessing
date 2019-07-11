#FROM alpine:3.1
FROM continuumio/miniconda3

ADD . /

#Creating an environment

RUN conda env create -f environment.yml

# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all

# Pull the environment name out of the environment.yml
RUN echo "source activate py36" > ~/.bashrc
ENV PATH /opt/conda/envs/py36/bin:$PATH

# JAVA
RUN tar -xvf jdk-8u211-linux-x64.tar.gz \
   && rm jdk-8u211-linux-x64.tar.gz

ENV JAVA_HOME /jdk1.8.0_211
ENV PATH JAVA_HOME/bin:$PATH

#SPARK
ENV SPARK_HOME /spark-2.4.3
ENV PATH $PATH:${SPARK_HOME}/bin
ENV PATH $PATH:${SPARK_HOME}/sbin

RUN wget http://mirrors.ircam.fr/pub/apache/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz \
    && tar -xvf spark-2.4.3-bin-hadoop2.7.tgz \
    && rm spark-2.4.3-bin-hadoop2.7.tgz

# HADOOP
ENV HADOOP_VERSION 2.7.1
ENV HADOOP_HOME /hadoop-$HADOOP_VERSION
ENV PATH $PATH:$HADOOP_HOME/bin

RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz \
  && tar -xvf hadoop-${HADOOP_VERSION}.tar.gz \
  && rm hadoop-${HADOOP_VERSION}.tar.gz

ENV PYTHONPATH ${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.7-src.zip

#RUN echo "export JAVA_HOME=/spark/jdk1.8.0_211" >> ~/.bashrc
#  && echo "export SPARK_HOME=/spark/spark-${SPARK_VERSION}" >> ~/.bashrc
#  && echo "export PATH=$PATH:$JAVA_HOME/bin" >> ~/.bashrc
#  && echo "export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.bashrc
#  && echo "export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH >> ~/.bashrc






# Bundle app source
EXPOSE  80
CMD ["python", "prediction.py"]
