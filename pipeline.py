import google.cloud.aiplatform as aip
from typing import NamedTuple

from kfp import compiler, dsl

from kfp.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component)

PROJECT_ID = "kubeflow-xxxxx"
PIPELINE_ROOT = "gs://gcs_data_store"

aip.init(project=PROJECT_ID, staging_bucket=PIPELINE_ROOT)

@component(
    base_image="jagadeeshj/autoformer:latest",
)
def data_ingest(
    args: dict,
    dataset_path: InputPath(),
    flag: str,
) -> NamedTuple(
    "Outputs",
    [
        ("output_one", Dataset),  # Return parameters
        ("output_two", Dataset),
    ],
):
    import argparse
    from data_provider.data_factory import data_provider

    splits = dataset_path.split("/")
    root_path, data_path = "/".join(splits[:-1]), splits[-1]
    args["root_path"] = root_path
    args["data_path"] = data_path

    parser = argparse.Namespace()   

    for key, value in args.items():
        setattr(parser, key, value)

    data_set, data_loader = data_provider(parser, flag)
    return (data_set, data_loader)

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage"]
)

# Get the content from the bucket.
def download_from_gcs(
    uri: str,
    project_id: str,
    dataset: OutputPath(),
    ):
    from google.cloud import storage

    no_prefix_uri = uri[len("gs://") :]
    splits = no_prefix_uri.split("/")
    bucket_name, path = splits[0], "/".join(splits[1:])
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    blob.download_to_filename(dataset)

@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="autoformer-pipeline-v2",
)
def pipeline():
    
    params={"data": "ETTh1", "embed": "timeF", "batch_size": 32, "freq": "h", "seq_len": 96, "label_len": 48, "pred_len": 96, "features": "M", "target": "OT", "num_workers": 10 }
    
    data_download_task = download_from_gcs(uri="gs://gcs_data_store/ETT/ETTh1.csv", project_id=PROJECT_ID).set_display_name("Download dataset")

    train_data_split = data_ingest(
        args=params,
        dataset_path = data_download_task.outputs["dataset"],
        flag="train",
    ).set_display_name("train_data_split")

    test_data_split = data_ingest(
        args=params,
        dataset_path = data_download_task.outputs["dataset"],
        flag="test",
    ).set_display_name("test_data_split")

    val_data_split = data_ingest(
        args=params,
        dataset_path = data_download_task.outputs["dataset"],
        flag="val",
    ).set_display_name("val_data_split")

compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="pipeline.json"
)
     
DISPLAY_NAME = "autoformer"
job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    # enable_caching=False,
    template_path="pipeline.json",
    pipeline_root=PIPELINE_ROOT
)

job.run()

