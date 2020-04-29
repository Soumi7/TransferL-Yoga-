FROM python:3.6-slim
COPY ./app_y.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./best1.h5 /deploy/
WORKDIR /deploy/
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app_y.py"]
