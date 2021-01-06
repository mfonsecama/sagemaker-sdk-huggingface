import logging

logger = logging.getLogger("sagemaker")

from sagemaker.estimator import Framework
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.s3 import S3Downloader

from huggingface.utils import validate_version_or_image_args, get_container_device, plot_result, download_model
from sagemaker import Session
import tarfile
import os


class HuggingFace(Framework):
    """Custom sagemaker-sdk estimator impelemntation of the HuggingFace libaries."""

    # FIXME: Sagemaker currently only supports images from private ecr not public ecr
    # _public_ecr_template_string = "public.ecr.aws/t6m7g5n4/huggingface-{type}:0.0.1-{device}-transformers{transformers_version}-datasets{datasets_version}"
    _ecr_template_string = "558105141721.dkr.ecr.eu-central-1.amazonaws.com/huggingface-{type}:0.0.1-{device}-transformers{transformers_version}-datasets{datasets_version}"

    def __init__(
        self,
        entry_point="train.py",
        source_dir=None,
        hyperparameters=None,
        py_version="py3",
        framework_version={"transformers": "4.1.1", "datasets": "1.1.3"},
        image_name=None,
        distributions=None,
        **kwargs,
    ):
        # validating framework_version and python version
        self.framework_version = framework_version
        self.py_version = py_version
        validate_version_or_image_args(self.framework_version, self.py_version)

        # checking for instance_type
        if "instance_type" in kwargs:
            self.instance_type = kwargs["instance_type"]
        else:
            self.instance_type = "local"

        # build ecr_uri
        self.image_uri = self._get_container_image("training")

        # using or create a sagemaker session
        if "sagemaker_session" in kwargs:
            self.sagemaker_session = kwargs["sagemaker_session"]
        else:
            self.sagemaker_session = Session()

        #  for distributed training
        #     if distribution is not None:
        #     instance_type = renamed_kwargs(
        #         "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
        #     )

        #     validate_smdistributed(
        #         instance_type=instance_type,
        #         framework_name=self._framework_name,
        #         framework_version=framework_version,
        #         py_version=py_version,
        #         distribution=distribution,
        #         image_uri=image_uri,
        #     )

        #     warn_if_parameter_server_with_multi_gpu(
        #         training_instance_type=instance_type, distribution=distribution
        #     )

        # if "enable_sagemaker_metrics" not in kwargs:
        #     # enable sagemaker metrics for PT v1.3 or greater:
        #     if self.framework_version and Version(self.framework_version) >= Version("1.3"):
        #         kwargs["enable_sagemaker_metrics"] = True

        super(HuggingFace, self).__init__(entry_point, source_dir, hyperparameters, image_uri=self.image_uri, **kwargs)

        # self.distribution = distribution or {}

    def upload_model_to_hub(self):
        return

    def download_model(self, local_path=".", unzip=False):
        os.makedirs(local_path, exist_ok=True)
        return download_model(
            model_data=self.model_data,
            local_path=local_path,
            sagemaker_session=self.sagemaker_session,
            model_dir=self.latest_training_job.name,
            unzip=unzip,
        )

    def fit(self, inputs=None, wait=True, logs="All", job_name=None, experiment_config=None):
        # if self.credentials:
        #     HfApi().create_repo(
        #     token=args.huggingface_token,
        #     name=args.hub_repo_name
        # )
        print("Starting training")
        super(HuggingFace, self).fit(inputs, wait, logs, job_name, experiment_config)
        print("uploading model to hub")
        self.download_model("model", True)

    def plot_result(self, metrics="all"):
        return plot_result(self, metrics)

    # def hyperparameters(self):
    # for distributed training
    #   return

    def _get_container_image(self, container_type):
        """return container image ecr url"""
        device = get_container_device(self.instance_type)
        image_uri = self._ecr_template_string.format(
            device=device,
            transformers_version=self.framework_version["transformers"],
            datasets_version=self.framework_version["datasets"],
            type=container_type,
        )
        if device == "gpu":
            image_uri = f"{image_uri}-cu110"
        return image_uri

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs,
    ):
        """returns Pytorch model from sagemaker-sdk since its none for HF implemented
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=requirements.txt#using-third-party-libraries
        there fore we has to include a `requirements.txt` to the code folder
        """
        # TODO: Not implemented
        return None
        # if "image_uri" not in kwargs:
        #     kwargs["image_uri"] = self._get_container_image("inference")

        # kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        # return PyTorchModel(
        #     self.model_data,
        #     role or self.role,
        #     entry_point or self._model_entry_point(),
        #     framework_version="1.6",
        #     py_version=self.py_version,
        #     source_dir=(source_dir or self._model_source_dir()),
        #     container_log_level=self.container_log_level,
        #     code_location=self.code_location,
        #     model_server_workers=model_server_workers,
        #     sagemaker_session=self.sagemaker_session,
        #     vpc_config=self.get_vpc_config(vpc_config_override),
        #     dependencies=(dependencies or self.dependencies),
        #     **kwargs
        # )

