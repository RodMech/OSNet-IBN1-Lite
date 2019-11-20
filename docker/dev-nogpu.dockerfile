FROM python:3.6-slim

RUN mkdir /opt/project

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /opt/project