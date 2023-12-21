FROM stone:stone

WORKDIR /data/www/stone

ADD ./stone ./stone
ADD ./app.py ./app.py
ADD ./setting.py ./setting.py
