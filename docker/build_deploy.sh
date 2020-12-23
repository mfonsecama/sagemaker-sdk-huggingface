#!/bin/bash

docker_name=huggingface-training
cpu_tag=0.0.1-cpu-transformers4.1.1-datasets1.1.3
gpu_tag=0.0.1-gpu-transformers4.1.1-datasets1.1.3-cu110
aws_region=eu-central-1
aws_account_id=558105141721
aws_profile=hf-sm

aws ecr-public create-repository --repository-name $docker_name --profile $aws_profile > /dev/null

if [[ "$1" = "gpu" ]]; then
    docker build -t $docker_name $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name:$gpu_tag . -f Dockerfile.gpu
else
    docker build -t $docker_name $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name:$cpu_tag . -f Dockerfile.cpu

fi


aws ecr-public get-login-password \
    --region $aws_region \
    --profile $aws_profile \
| docker login \
    --username AWS \
    --password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com


docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name