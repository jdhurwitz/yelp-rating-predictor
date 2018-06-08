FROM python:3.6.5

#set working dir to the bcn dir
COPY . /app
WORKDIR /app

#Install needed pckages
RUN pwd
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#expose port 80
EXPOSE 80

