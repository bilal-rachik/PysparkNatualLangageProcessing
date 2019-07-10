#FROM alpine:3.1
FROM continuumio/miniconda3

#Creating an environment
ADD . /
RUN conda env create -f environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc

ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

# Bundle app source
EXPOSE  80
CMD ["python", "prediction.py"]
