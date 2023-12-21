FROM stone:base

WORKDIR /data/www/stone

ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir -p /data/download