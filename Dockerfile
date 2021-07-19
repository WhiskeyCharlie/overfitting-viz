FROM python:3.8-buster


COPY ML-Overfitting-Plotly-master/assets /assets
COPY ML-Overfitting-Plotly-master/*.py /
COPY ML-Overfitting-Plotly-master/requirements.txt /


RUN pip install -r requirements.txt


#EXPOSE 2522

CMD ["python", "app.py"]
