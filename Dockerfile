# FB prophet
# FROM gcr.io/silicon-badge-274423/stock-predictions

# ARG CHUNK_NUMBER

# ENV CHUNK_NUMBER=$CHUNK_NUMBER

# COPY . .

# lstm_model_price_features_vol_v4
FROM tensorflow/tensorflow:latest-gpu

ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

RUN apt-get update
RUN apt-get -y install vim
RUN pip install pandas
RUN pip install google-cloud-bigquery
RUN pip install tensorflow
RUN pip install pyarrow
RUN pip install google-cloud-bigquery-storage
RUN pip install sklearn

COPY . .

CMD [ "python", "lstm_model_price_features_vol_v8.py" ]

# lstm_model_price_features_vol_v5
# FROM tensorflow/tensorflow:latest-gpu
# ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

# RUN apt-get update
# RUN apt-get -y install vim
# RUN pip install pandas
# RUN pip install google-cloud-bigquery
# RUN pip install tensorflow
# RUN pip install pyarrow
# RUN pip install google-cloud-bigquery-storage
# RUN pip install sklearn

# COPY . .

# CMD [ "python", "lstm_model_price_features_vol_v5.py" ]

# boosted_tree_features_vol_v4
# FROM python:3.8.5
# ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
# RUN apt-get update
# RUN apt-get -y install vim
# RUN pip install pandas
# RUN pip install google-cloud-bigquery
# RUN pip install tensorflow
# RUN pip install pyarrow
# RUN pip install google-cloud-bigquery-storage
# RUN pip install -U --user pip numpy wheel
# RUN pip install sklearn
# COPY . .
# CMD [ "python", "boosted_tree_features_vol_v4.py" ]
