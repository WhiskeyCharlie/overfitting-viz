FROM python:3.8-buster


COPY ML-Overfitting-Plotly-master/assets /assets
COPY ML-Overfitting-Plotly-master/*.py /
COPY ML-Overfitting-Plotly-master/requirements.txt /


RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 2522

CMD ["gunicorn", "-b", "0.0.0.0:2522", "-w", "2", "--threads", "2", "app:server"]
