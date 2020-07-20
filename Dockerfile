# docker build --tag us.gcr.io/bedrockdata-qa/sync:latest --rm --file deploy/sync.Dockerfile .
# docker build --tag compopt:latest .
FROM python:3.6.5

WORKDIR /usr/src/app
#COPY . /usr/src/app/

COPY ./scripts/main.py /usr/src/app/main.py
COPY ./scripts/compostLP.py /usr/src/app/compostLP.py
COPY ./data /usr/src/app/data
COPY ./requirements.txt /usr/src/app/requirements.txt

#COPY ./src /usr/src/app/src
#COPY ./index.js /usr/src/app/index.js


# Install strace for healthcheck script
# RUN apt-get install strace

# Install python libraries
# RUN pip install -r dev-requirements.txt
RUN pip install -r requirements.txt

# Add bedrock_sync to python path
# ENV PYTHONPATH "{$PYTHONPATH}:/usr/src/app/"


# Run syncd in production mode
CMD ["python", "main.py"]

# CMD tail -f /dev/null