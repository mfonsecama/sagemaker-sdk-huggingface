# Getting Started

Here you can build the different HuggingFace Sagemaker container. You can build

for training

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

for inference

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

## Script Arguments

You can pass mutliple named arguments to the script.

```bash
--image # The container image type either training, inference | default training
--device # The container device either cpu, gpu or test | default cpu
--account_id # The account_id of the aws account/registry | default 558105141721
--profile # The aws profile which going to be used | default default for CI-Pipelines use ci
--transformers_version # The transformers version which should be used in the container | default 4.1.1
--datasets_version # The datasets version which should be used in the container | default 1.1.3
--version # The Container version | default 0.0.1
```

**usage**

```bash
./build_push.sh --device gpu --type training  --version 1.0.0
```

## Build and Push Container

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
