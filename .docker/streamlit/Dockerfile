FROM python:3.10.11

WORKDIR /app
COPY . /app

RUN pip install -U kaleido
RUN pip install -r ./requirements.txt

CMD ["sh", "-c", "streamlit run app.py --server.port 80"]