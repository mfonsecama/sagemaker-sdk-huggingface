# sagemaker-sdk-huggingface

https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/custom-training-containers/script-mode-container-2/notebook/script-mode-container-2.ipynb

https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework

---

# HugginFace Deep Learning Container

Here you can build the different HuggingFace Sagemaker container. You can build

for training

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

for inference

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

### Outline

- [Container Image List](#quick-start)
- [Script Arguments](#script-args)
- [Build and Push Container Example](#xample)

# 🔮 <a name="container-list"></a>Container Image List

| Framework | type      | device | base                                            | python-version | transformers-version | datasets-version | URL                                                                                      |
| --------- | --------- | ------ | ----------------------------------------------- | -------------- | -------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| 0.0.1     | training  | cpu    | aws dlc pytorch1.6.0-cpu-py36-ubuntu16.04       | 3.6.10         | 4.1.1                | 1.1.3            | `public.ecr.aws/t6m7g5n4/huggingface-training:0.0.1-cpu-transformers4.1.1-datasets1.1.3` |
| 0.0.1     | training  | gpu    | aws dlc pytorch1.6.0-gpu-py36-cu110-ubuntu16.04 | 3.6.10         | 4.1.1                | 1.1.3            |                                                                                          |
| 0.0.1     | inference | cpu    | aws dlc pytorch1.6.0-cpu-py36-ubuntu16.04       | 3.6.10         | 4.1.1                | 1.1.3            |                                                                                          |
| 0.0.1     | inference | gpu    | aws dlc pytorch1.6.0-gpu-py36-cu110-ubuntu16.04 | 3.6.10         | 4.1.1                | 1.1.3            |                                                                                          |

# ⚙️ <a name="script-args"></a> Script Arguments

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
./build_push.sh --device gpu --type training  --version 1.0.0
```

# 🏗 <a name="example"></a> Build and Push Container Example

**GPU Container Training**

```bash
./build_push.sh --device gpu --type training
```

**GPU Container Inference**

```bash
./build_push.sh --device gpu --type inference
```

**CPU Container Training**

```bash
./build_push.sh --device cpu --type training
```

**CPU Container Inference**

```bash
./build_push.sh --device cpu --type inference
```
