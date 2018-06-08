FROM python:3.6

#set working dir to the bcn dir
ADD ./bcn_based-master

#Install needed pckages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#expose port 80
EXPOSE 80

