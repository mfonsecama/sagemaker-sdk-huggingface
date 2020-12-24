from sagemaker.estimator import Framework
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.s3.S3Downloader import download as download_from_s3 , read_file as read_file_from_s3
import re


class HuggingFace(Framework):
    
    _ecr_template_string = "public.ecr.aws/t6m7g5n4/huggingface-training:0.0.1-{device}-transformers{transformers_version}-datasets{datasets_version}"
    
    def __init__(
        self,
        entry_point,
        source_dir=None,
        hyperparameters=None,
        py_version="py3",
        framework_version={'transformers':'4.1.1','datasets':'1.1.3'},
        image_name=None,
        distributions=None,
        **kwargs
    ):
      
        self.image_name = self._get_container_image()
      
        super(CustomFramework, self).__init__(
            entry_point, source_dir, hyperparameters, image_name=self.image_name, **kwargs
        )
    
    def _configure_distribution(self, distributions):
        return
    
    def upload_model_to_hub(self):
        return

    def download_model(self):
        download_from_s3(self.model_data)
    
    # def hyperparameters(self):
    # for distributed training
    #   return
    
    def _get_container_image(self):
      """return container image ecr url"""
      device = self._get_image_typ()
      return _ecr_template_string.format(device=device,transformers_version=self.framework_version['transformers'],datasets_version=self.framework_version['datasets'])

    def _get_container_device(self):
      """identifies container device """
      if self.instance_type.startswith("local"):
              device = "cpu" if self.instance_type == "local" else "gpu"
      else:
          # looks for either "ml.<family>.<size>" or "ml_<family>"
          match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", self.instance_type)
          if match:
              family = match[1]
              # 'cpu' or 'gpu'.
              elif family.startswith("inf"):
                  processor = "inf"
              elif family[0] in ("g", "p"):
                  processor = "gpu"
              else:
                  processor = "cpu"
          else:
              raise ValueError(
                  "Invalid SageMaker instance type: {}. For options, see: "
                  "https://aws.amazon.com/sagemaker/pricing/instance-types".format(self.instance_type)
              )
    return device
    
    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs
    ):
        """returns Pytorch model from sagemaker-sdk since its none for HF implemented
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=requirements.txt#using-third-party-libraries
        there fore we has to include a `requirements.txt` to the code folder
        """
        
        return PyTorchModel()