FROM python:3.7-slim

ENV ECS_TASK=true

WORKDIR /usr/local/ml
COPY pipelines ./pipelines
COPY scripts/start_pipeline.sh .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["/bin/bash","start_pipeline.sh"]