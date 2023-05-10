# Sport traffic prediction using AWS Sagemaker

This project presents a code for building Amazon SageMaker Pipelines (**ml-pipelines**) to create end-to-end workflows that manage and deploy SageMaker jobs aimed to train models and predict time series of hockey events requests traffic with 10 days horizon

The project also contains additional examples of **data-ingestion** steps and **postprocessing** steps which supposed to be managed separately.

The e2e DAG is composed in MWAA (Amazon Airflow) and simplified schema is below

![e2e prediction dag](e2e.jpeg)
