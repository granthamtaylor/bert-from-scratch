FROM python:3.10-buster

WORKDIR /app

RUN \
  pip install --upgrade pip && \
  apt update && \
  apt -y install entr && \
  touch /trigger.txt

ADD requirements.txt /requirements.txt
RUN \
  chmod -R 777 /requirements.txt && \
  pip install -r /requirements.txt

ADD ./src/ /app/
