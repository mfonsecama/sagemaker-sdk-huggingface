import re
from sagemaker.s3 import S3Downloader
import tarfile
import os


def validate_version_or_image_args(framework_version, py_version):
    """Checks if libary or python version arguments are specified correct.
    Args:
        framework_version (dict): Dictonary with 'transformers' and 'datasets' as key and their version of the library.
        py_version (str): The version of Python.
    Raises:
        ValueError: if `framework_version['transfomers']` is None and `framework_version['datasets']` is None and `py_version` is
            not 3.
    """
    if (
        framework_version is None
        or py_version != "py3"
        or "transformers" not in framework_version
        or "datasets" not in framework_version
    ):
        raise ValueError(
            "framework_version or py_version was None, "
            "Either specify both framework_version and py_version, or specify image_uri."
        )


def get_container_device(instance_type):
    """identifies container device """
    if instance_type.startswith("local"):
        device = "cpu" if instance_type == "local" else "gpu"
    elif re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type):
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        family = match[1]
        # 'cpu' or 'gpu'.
        if family.startswith("inf"):
            device = "inf"
        elif family[0] in ("g", "p"):
            device = "gpu"
        else:
            device = "cpu"
    else:
        raise ValueError(
            "Invalid SageMaker instance type: {}. For options, see: "
            "https://aws.amazon.com/sagemaker/pricing/instance-types".format(instance_type)
        )
    return device


def download_model(model_data, local_path=".", unzip=False, sagemaker_session=None, model_dir="model"):
    """Downloads model file from sagemaker training to local directory and unzips its to directory if wanted."""
    S3Downloader.download(
        s3_uri=model_data, local_path=os.path.join(local_path, model_dir), sagemaker_session=sagemaker_session
    )
    if unzip:
        with tarfile.open(os.path.join(local_path, model_dir, "model.tar.gz"), "r:gz") as model_zip:
            model_zip.extractall(path=os.path.join(local_path, model_dir))
        os.remove(os.path.join(local_path, model_dir, "model.tar.gz"))
