import logging
import os
import sys
from typing import Dict, Optional

from sagemaker import Session
from sagemaker.estimator import Framework
from transformers.hf_api import HfApi

from huggingface.utils import (
    HfRepository,
    download_model,
    get_container_device,
    plot_results,
    validate_version_or_image_args,
)


logger = logging.getLogger("sagemaker")

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class HuggingFace(Framework):
    """
    Custom sagemaker-sdk estimator impelemntation of the HuggingFace libaries. This implementation is oriented towards the Pytorch implementation.
    https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/pytorch/estimator.py
    """

    # FIXME: Sagemaker currently only supports images from private ecr not public ecr
    # _public_ecr_template_string = "public.ecr.aws/t6m7g5n4/huggingface-{type}:0.0.1-{device}-transformers{transformers_version}-datasets{datasets_version}"
    _ecr_template_string = "558105141721.dkr.ecr.eu-central-1.amazonaws.com/huggingface-{type}:0.0.1-{device}-transformers{transformers_version}-datasets{datasets_version}"

    def __init__(
        self,
        entry_point="train.py",
        source_dir: str = None,
        hyperparameters: Optional[Dict[str, str]] = None,
        framework_version={"transformers": "4.1.1", "datasets": "1.1.3"},
        image_uri: Optional[str] = None,
        huggingface_token: str = None,
        # distribution=None,
        **kwargs,
    ):
        """
        Args:
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for
                training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            framework_version (dict): Transformers and datasets versions you want to use for
                executing your model training code. Defaults to ``{"transformers": "4.1.1", "datasets": "1.1.3"}``. Required unless
                ``image_uri`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#pytorch-sagemaker-estimators.
            image_uri (str): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.
                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``
                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            huggingface_token (str): HuggingFace Hub authentication token for uploading your model files.
                You can get this by either using the [transformers-cli](https://huggingface.co/transformers/model_sharing.html) with `transfomers-cli login`
                or using the `login()` method of `transformers.hf_api`. If the HuggingFace Token is provided the model will uploaded automatically to the
                model hub using the `base_job_name` as repository name.
        """
        # validating framework_version and python version
        self.framework_version = framework_version
        self.py_version = "py3"
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

        super(HuggingFace, self).__init__(entry_point, source_dir, hyperparameters, image_uri=self.image_uri, **kwargs)

        if huggingface_token:
            logger.info(
                f"estimator initialized with HuggingFace Token, model will be uploaded to hub using the {self.base_job_name} as repostiory name"
            )
        self.huggingface_token = huggingface_token

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
        if self.huggingface_token and wait is True:
            logger.info(f"creating repository {self.base_job_name} on the HF hub")
            self.repo_url = HfApi().create_repo(token=self.huggingface_token, name=self.base_job_name)

        # parent fit method
        super(HuggingFace, self).fit(inputs, wait, logs, job_name, experiment_config)

        if self.huggingface_token and wait is True:
            logger.info(f"downloading model to {self.latest_training_job.name}/ ")
            self.download_model(".", True)

            logger.info(f"initalizing model repository ")
            model_repo = HfRepository(
                repo_url=self.repo_url,
                huggingface_token=self.huggingface_token,
                model_dir=f"./{self.latest_training_job.name}",
            )

            logger.info("uploading model files to HF hub")
            model_repo.commit_files_and_push_to_hub()

    def plot_result(self, metrics="all"):
        return plot_results(self, metrics)

    def _get_container_image(self, container_type: str) -> str:
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
        """returns no model from sagemaker-sdk since its none for HF implemented
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=requirements.txt#using-third-party-libraries
        there fore we has to include a `requirements.txt` to the code folder
        """
        raise NotImplementedError
