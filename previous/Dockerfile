FROM python:3.12-slim

# USER root
WORKDIR /app
#Install requirements cached before everything
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
# RUN chmod u+x app.py  
EXPOSE 7860

CMD [ "python","app.py" ]
