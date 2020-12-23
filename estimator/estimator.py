from sagemaker.estimator import Framework
import re


class HuggingFace(Framework):
    def __init__(
        self,
        entry_point,
        source_dir=None,
        hyperparameters=None,
        py_version="py3",
        framework_version=None,
        image_name=None,
        distributions=None,
        **kwargs
    ):
      
        self.image_name = self._get_container_image()
      
        super(CustomFramework, self).__init__(
            entry_point, source_dir, hyperparameters, image_name=image_name, **kwargs
        )
    
    def _configure_distribution(self, distributions):
        return
    
    def upload_model_to_hub(self):
      return
    
    def hyperparameters(self):
      return
    
    def _get_container_image(self):
      """"""
      return

    def _get_image_typ(self,instance_type):
      """identifies container image"""
      if instance_type.startswith("local"):
              processor = "cpu" if instance_type == "local" else "gpu"
      else:
          # looks for either "ml.<family>.<size>" or "ml_<family>"
          match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
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
                  "https://aws.amazon.com/sagemaker/pricing/instance-types".format(instance_type)
              )
    return processor
    
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
        return None