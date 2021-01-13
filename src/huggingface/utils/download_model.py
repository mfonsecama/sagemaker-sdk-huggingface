import os
import re
import tarfile

from sagemaker.s3 import S3Downloader


def download_model(model_data, local_path=".", unzip=False, sagemaker_session=None, model_dir="model"):
    """Downloads model file from sagemaker training to local directory and unzips its to directory if wanted."""
    S3Downloader.download(
        s3_uri=model_data, local_path=os.path.join(local_path, model_dir), sagemaker_session=sagemaker_session
    )
    if unzip:
        with tarfile.open(os.path.join(local_path, model_dir, "model.tar.gz"), "r:gz") as model_zip:
            model_zip.extractall(path=os.path.join(local_path, model_dir))
        os.remove(os.path.join(local_path, model_dir, "model.tar.gz"))
