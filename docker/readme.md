# Getting Started

Here you can build the different HuggingFace Sagemaker container. You can build

- a gpu container based on AWS DLC Pytorch1.6
- a cpu container based on AWS DLC Pytorch1.6
- a test container just checking if the parameters are passed correctly

## Script Arguments

You can pass mutliple named arguments to the script.

```bash
--type # The container type either cpu, gpu or test | default cpu
--account_id # The account_id of the aws account/registry | default 558105141721
--profile # The aws profile which going to be used | default default for CI-Pipelines use ci
--transformers_version # The transformers version which should be used in the container | default 4.1.1
--datasets_version # The datasets version which should be used in the container | default 1.1.3
--version # The Container version | default 0.0.1
```

**usage**

```bash
./build_push.sh --type gpu --version 1.0.0
```

## Build and Push Container

**GPU Container**

```bash
./build_push.sh --type gpu
```

**CPU Container**

```bash
./build_push.sh --type cpu
```
