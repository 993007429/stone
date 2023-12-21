FROM stone:stone

WORKDIR /data/www/stone

ADD ./stone ./stone
ADD ./app.py ./app.py
ADD ./setting.py ./setting.py
ADD ./local_settings ./local_settings
ADD ./model_versions ./model_versions
