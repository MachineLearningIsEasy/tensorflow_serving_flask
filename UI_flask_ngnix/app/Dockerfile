FROM tiangolo/uwsgi-nginx:python3.6

COPY ./ /app

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
