FROM python:3.8.5

WORKDIR /models
ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "fb_prophet.py" ]
