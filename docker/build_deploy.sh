#!/bin/bash
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu110-ubuntu16.04	
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-cpu-py36-ubuntu16.04	


docker_name=huggingface-training
cpu_tag=0.0.1-cpu-transformers4.1.1-datasets1.1.3
gpu_tag=0.0.1-gpu-transformers4.1.1-datasets1.1.3-cu110
aws_region=eu-central-1
aws_account_id=891511646143
aws_profile=serverless-bert

aws ecr create-repository --repository-name $docker_name --profile $aws_profile > /dev/null


if $1 cpu 
    docker build -t $docker_name $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name:$cpu_tag . -f 
else if $1 gpu
    docker build -t $docker_name $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name:$cpu_tag . -f 

aws ecr get-login-password \
    --region $aws_region \
    --profile $aws_profile \
| docker login \
    --username AWS \
    --password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com


docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/$docker_name