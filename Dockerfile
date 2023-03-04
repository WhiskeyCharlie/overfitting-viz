FROM python:3.10-slim-bullseye


COPY requirements.txt /
COPY src/*.py /app/
COPY assets /assets

EXPOSE 2522
ENV IP_TO_LISTEN_ON="0.0.0.0"

ARG NUM_WORKERS=3
ARG NUM_THREADS_PER_WORKER=1

ENV NUM_WORKERS ${NUM_WORKERS}
ENV NUM_THREADS_PER_WORKER ${NUM_THREADS_PER_WORKER}

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir gunicorn

CMD gunicorn -b "0.0.0.0:2522" -w ${NUM_WORKERS} --threads ${NUM_THREADS_PER_WORKER} --chdir "app" "app:server"
