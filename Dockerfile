FROM python:3.8-buster



COPY src/*.py /app/
COPY assets /assets
COPY requirements.txt /


RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 2522

CMD ["gunicorn", "-b", "0.0.0.0:2522", "-w", "2", "--threads", "2", "--chdir", "app", "app:server"]
