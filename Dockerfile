FROM gcr.io/silicon-badge-274423/stock-predictions

ARG CHUNK_NUMBER

ENV CHUNK_NUMBER=$CHUNK_NUMBER

COPY . .

CMD [ "python", "fb_prophet.py" ]
