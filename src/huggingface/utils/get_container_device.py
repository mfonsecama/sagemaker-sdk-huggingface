import re


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
