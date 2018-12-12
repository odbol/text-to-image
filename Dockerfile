FROM tensorflow/tensorflow:latest-gpu

RUN pip install --upgrade tensorlayer
RUN pip install -U nltk