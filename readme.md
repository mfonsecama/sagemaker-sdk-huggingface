# 🤗 sagemaker-sdk-huggingface

This repostiory implements a cutsom sagemaker-sdk extension for the HuggingFace libraries. The repostoriy is split into 3 parts. First there is `docker/`, which contains the `dockerfiles` and scripts to create the AWS DLC for HuggingFace. Second there is `hugginface/`, which includes the custom `HuggingFace()` extension for the sagemaker-sdk. Lastly there is another folder `examples/`, which contains multiple examples on how to use the `HuggingFace()` extension for sagemaker-sdk

This Repository contains multiple examples "how to use the transformers and datasets library from HuggingFace with AWS Sagemaker". All Notebooks can be run locally or within AWS Sagemaker Studio.

### Outline

- [Container Image List](#container-list)
- [Script Arguments](#script-args)
- [Build and Push Container Example](#example)
- [How to use HuggingFace sagemaker-sdk extension](#sdk)
- [Example Overview](#exov)

---

# 🧱 HugginFace Deep Learning Container

You can build different HuggingFace Deep Learning Container to use them in AWS Sagemaker.

for training

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

for inference

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

## 🔮 <a name="container-list"></a>Container Image List

_**NOTE:** Sadly Sagemaker doesn´t support `public.ecr.aws` images so therefore, we have to used container in `private` registry. To use a private registry you can use the `docker/build_push_private_ecr.sh` script to build and push the container to your private ecr registry._

| type      | device | base                                            | python-version | transformers-version | datasets-version | public-URL                                                                                     |
| --------- | ------ | ----------------------------------------------- | -------------- | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| training  | cpu    | aws dlc pytorch1.6.0-cpu-py36-ubuntu16.04       | 3.6.10         | 4.1.1                | 1.1.3            | `public.ecr.aws/t6m7g5n4/huggingface-training:0.0.1-cpu-transformers4.1.1-datasets1.1.3`       |
| training  | gpu    | aws dlc pytorch1.6.0-gpu-py36-cu110-ubuntu16.04 | 3.6.10         | 4.1.1                | 1.1.3            | `public.ecr.aws/t6m7g5n4/huggingface-training:0.0.1-gpu-transformers4.1.1-datasets1.1.3-cu110` |
| inference | cpu    | aws dlc pytorch1.6.0-cpu-py36-ubuntu16.04       | 3.6.10         | 4.1.1                | 1.1.3            |                                                                                                |
| inference | gpu    | aws dlc pytorch1.6.0-gpu-py36-cu110-ubuntu16.04 | 3.6.10         | 4.1.1                | 1.1.3            |

## ⚙️ <a name="script-args"></a> Script Arguments

You can pass mutliple named arguments to the script.

| parameter              | default      | description                                                        |
| ---------------------- | ------------ | ------------------------------------------------------------------ |
| --image_type           | training     | The container image type either training, inference                |
| --device               | cpu          | The container device either cpu, gpu or test                       |
| --account_id           | 558105141721 | The aws account_id of the aws account/registry                     |
| --profile              | default      | The aws profile which going to be used. Pass `ci` for CI-Pipelines |
| --transformers_version | 4.1.1        | The transformers version which will be installed in the container  |
| --datasets_version     | 1.1.3        | The datasets version which will be installed in the container      |
| --version              | 0.0.1        | The container version                                              |

**usage**

```bash
./docker/build_push_private_ecr.sh --device gpu --type training  --version 1.0.0
```

## 🏗 <a name="example"></a> Build and Push Container Example

Since public.ecr is not supported by sagemaker currently you have to build the docker image for yourself and upload it to your private ecr registry.

**GPU Container Training**

```bash
cd docker && ./build_push_private_ecr.sh --device gpu --image_type training --profile hf-sm
```

**GPU Container Inference**

```bash
./docker/build_push_private_ecr.sh --device gpu --image_type inference
```

**CPU Container Training**

```bash
./docker/build_push_private_ecr.sh --device cpu --image_type training
```

**CPU Container Inference**

```bash
./docker/build_push_private_ecr.sh --device cpu --image_type inference
```

# 🧵 <a name="sdk"></a> How to use HuggingFace sagemaker-sdk extension

This Repository contains multiple examples "how to use the transformers and datasets library from HuggingFace with AWS Sagemaker". All Notebooks can be run locally or within AWS Sagemaker Studio.

**example strucute**

Each folder starting with `0X_...` contains an sagemaker example.  
Each example contains a jupyter notebook `sagemaker-example.ipynb`, which is used to start train job on AWS Sagemaker or preprocess data.

As explained above, you are able to run these examples either on your local machine or in the AWS Sagemaker Studio.

## <a name="exov"></a> Example Overview

| example                                                                                                                                                                                            | description                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [01_basic_example_huggingface_extension](https://github.com/philschmid/sagemaker-sdk-huggingface/blob/main/examples/01_basic_example_huggingface_extension/sagemaker-notebook.ipynb)               | This example uses the custom HuggingFace sagemaker extension. In the fine-tuning scripts, it uses the `Trainer` class. The dataset is processed in jupyter notebook with the `datasets` library and then uploaded to S3. |
| [02_spot_instances_with_huggingface_extension](https://github.com/philschmid/sagemaker-sdk-huggingface/blob/main/examples/02_spot_instances_with_huggingface_extension/sagemaker-notebook.ipynb)   | It is the same example as `01_basic_example_huggingface_extension`, but we will use ec2 spot instances for training, which can reduce the training cost up to 90%                                                        |
| [03_track_custom_metrics_huggingface_extension](https://github.com/philschmid/sagemaker-sdk-huggingface/blob/main/examples/03_track_custom_metrics_huggingface_extension/sagemaker-notebook.ipynb) | It is the same example as `02_spot_instances_with_huggingface_extension`, but we will use `custom_metrics` to track validation metrics in our training job and plot them into the notebook                               |

## Getting started locally

If you want to use an example on your local machine, you need:

- an AWS Account
- configured AWS credentials on your local machine,
- an AWS Sagemaker IAM Role

If you don´t have an AWS account you can create one [here](https://portal.aws.amazon.com/billing/signup?nc2=h_ct&src=header_signup&redirect_url=https%3A%2F%2Faws.amazon.com%2Fregistration-confirmation#/start). To configure AWS credentials on your local machine you can take a look [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Lastly, to create an AWS Sagemaker IAM Role you can take a look [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html), beaware if you change the name of the role, you have to adjust it in the jupyter notebook. Now you have to install dependencies from the `requirements.txt` and you are good to go.

```bash
pip install -r requirements.txt
```

## Getting started with Sagemaker Studio

If you want to use an example in sagemaker studio. You can open your sagemaker studio and then clone the github repository. Afterwards you have to install dependencies from the `requirements.txt`.

# Troubleshoot

- If you get an `UnknownServiceError` with `Unknown service: 'sagemaker-featurestore-runtime'` run `pip install -r requirements.txt --upgrade` and restart your jupyter runtime.

Links:
https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/custom-training-containers/script-mode-container-2/notebook/script-mode-container-2.ipynb

https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework

[pytorch inference example](https://github.com/aws/amazon-sagemaker-examples/tree/master/frameworks/pytorch/code)
