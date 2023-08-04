import google.cloud.aiplatform as aip

from kfp import compiler, dsl

from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    InputPath,
    Model,
    Output,
    OutputPath,
    component,
)

# Update below configs before running the pipeline
PROJECT_ID = "kubeflow-394503"
PIPELINE_ROOT = "gs://gcs_data_store"
BASE_IMAGE = "jagadeeshj/autoformer:v2"
LOCATION = "us-central1-a"

aip.init(
    project=PROJECT_ID,
    staging_bucket=PIPELINE_ROOT,
)


# Get the content from the bucket.
@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage"],
)
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


@component(
    base_image=BASE_IMAGE,
)
def data_ingest(
    args: dict,
    dataset_path: InputPath(),
    flag: str,
    dataset: OutputPath(),
    dataloader: OutputPath(),
):
    import torch
    import argparse
    from data_provider.data_factory import (
        data_provider,
    )

    splits = dataset_path.split("/")
    root_path, data_path = (
        "/".join(splits[:-1]),
        splits[-1],
    )
    args["root_path"] = root_path
    args["data_path"] = data_path

    parser = argparse.Namespace()

    for key, value in args.items():
        setattr(parser, key, value)

    data_set, data_loader = data_provider(parser, flag)
    torch.save(data_set, dataset)
    torch.save(data_loader, dataloader)

    return


@component(
    base_image=BASE_IMAGE,
)
def train(
    train_data: InputPath(),
    train_loader: InputPath(),
    vali_data: InputPath(),
    vali_loader: InputPath(),
    test_data: InputPath(),
    test_loader: InputPath(),
    args: dict,
    model_out: OutputPath(),
):
    import torch
    import argparse
    from exp.exp_main import Exp_Main

    parser = argparse.Namespace()

    for key, value in args.items():
        setattr(parser, key, value)

    train_data = torch.load(train_data)
    train_loader = torch.load(train_loader)
    vali_data = torch.load(vali_data)
    vali_loader = torch.load(vali_loader)
    test_data = torch.load(test_data)
    test_loader = torch.load(test_loader)

    exp = Exp_Main(parser)
    exp.train(
        train_data,
        train_loader,
        vali_data,
        vali_loader,
        test_data,
        test_loader,
        model_out,
    )

    return


@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="autoformer-pipeline-v2",
)
def pipeline():
    basic_config = {"is_training": 1, "model_id": "test", "model": "Autoformer"}
    data_loader = {
        "data": "ETTh1",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "checkpoints": "./checkpoints/",
    }
    forecasting_task = {"seq_len": 96, "label_len": 48, "pred_len": 96}
    model = {
        "bucket_size": 4,
        "n_hashes": 4,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.05,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "do_predict": True,
    }
    optimization = {
        "num_workers": 10,
        "itr": 2,
        "train_epochs": 1,
        "batch_size": 2,
        "patience": 3,
        "learning_rate": 0.0001,
        "des": "test",
        "loss": "mse",
        "lradj": "type1",
        "use_amp": False,
    }
    gpu = {
        "use_gpu": False,
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
    }

    params = {
        **basic_config,
        **data_loader,
        **forecasting_task,
        **model,
        **optimization,
        **gpu,
    }

    data_download_task = download_from_gcs(
        uri="gs://gcs_data_store/ETT/ETTh1.csv",
        project_id=PROJECT_ID,
    ).set_display_name("Download dataset")

    train_data_split = (
        data_ingest(
            args=params,
            dataset_path=data_download_task.outputs["dataset"],
            flag="train",
        )
        .after(data_download_task)
        .set_display_name("train_data_split")
    )

    test_data_split = (
        data_ingest(
            args=params,
            dataset_path=data_download_task.outputs["dataset"],
            flag="test",
        )
        .after(data_download_task)
        .set_display_name("test_data_split")
    )

    val_data_split = (
        data_ingest(
            args=params,
            dataset_path=data_download_task.outputs["dataset"],
            flag="val",
        )
        .after(data_download_task)
        .set_display_name("val_data_split")
    )

    train_task = (
        train(
            train_data=train_data_split.outputs["dataset"],
            train_loader=train_data_split.outputs["dataloader"],
            vali_data=val_data_split.outputs["dataset"],
            vali_loader=val_data_split.outputs["dataloader"],
            test_data=test_data_split.outputs["dataset"],
            test_loader=test_data_split.outputs["dataloader"],
            args=params,
        )
        .after(train_data_split)
        .set_display_name("train_task")
        # https://cloud.google.com/vertex-ai/docs/pipelines/machine-types
        .add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_K80')
        .set_gpu_limit(1)
        )


compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="pipeline.json",
)

DISPLAY_NAME = "autoformer"
job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    # enable_caching=False,
    template_path="pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    location=LOCATION
)

job.run()
