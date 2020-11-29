# FB prophet
# FROM gcr.io/silicon-badge-274423/stock-predictions

# ARG CHUNK_NUMBER

# ENV CHUNK_NUMBER=$CHUNK_NUMBER

# COPY . .

FROM lppier/docker-prophet

ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

RUN apt-get update
RUN apt-get -y install vim
RUN pip install google-cloud-bigquery
RUN pip install sklearn
RUN pip install tensorflow
RUN pip install pyarrow
RUN pip install google-cloud-bigquery-storage

COPY . .

CMD [ "python", "lstm_returns.py" ]
