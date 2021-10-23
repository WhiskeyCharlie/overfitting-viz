FROM python:3.8-slim-buster


COPY src/*.py /app/
COPY assets /assets
COPY requirements.txt /


RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 2522
ENV IP_TO_LISTEN_ON="0.0.0.0"

ARG NUM_WORKERS=3
ARG NUM_THREADS_PER_WORKER=1

ENV NUM_WORKERS ${NUM_WORKERS}
ENV NUM_THREADS_PER_WORKER ${NUM_THREADS_PER_WORKER}

CMD gunicorn -b "0.0.0.0:2522" -w ${NUM_WORKERS} --threads ${NUM_THREADS_PER_WORKER} --chdir "app" "app:server"
