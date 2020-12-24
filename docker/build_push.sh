#!/bin/bash

# huggingface versions
transformers_version=4.1.1
datasets_version=1.1.3

# docker variables
container_name=huggingface-training
container_type=null

# container versions/tags
version=0.0.1
cpu_tag=$version-cpu-transformers$transformers_version-datasets$datasets_version
gpu_tag=$version-gpu-transformers$transformers_version-datasets$datasets_version-cu110

# aws parameters
aws_account_id=558105141721
aws_profile=hf-sm

#registry parameters
ecr_url=public.ecr.aws
ecr_alias=t6m7g5n4


# parsing named arguments
while (( $# > 1 )); do case $1 in
   --type) container_type="$2";;
   --account_id) aws_account_id="$2";;
   --transformers_version) transformers_version="$2";;
   --datasets_version) datasets_version="$2";;
   --version) version="$2";;
   *) break;
 esac; shift 2
done

# can be run only once! public repositories has to be created in us-east-1
# aws ecr-public create-repository --repository-name $container_name  --profile $aws_profile --region us-east-1   > /dev/null

# Starting print 
echo "####################### Build and Push HuggingFace Sagemaker Container #######################"
echo ""
echo "Build parameters:"
echo "container_type=$container_type"
echo "aws_account_id=$aws_account_id"
echo "transformers_version=$transformers_version"
echo "datasets_version=$datasets_version"
echo "container_version=$version"

if [[ $container_type = "gpu" ]]; then
    echo "Building gpu container...."
    tag=$gpu_tag
    dockerfile=Dockerfile.gpu
elif [[ $container_type = "cpu" ]]; then
    echo "Building cpu container...."
    tag=$cpu_tag
    dockerfile=Dockerfile.cpu
elif [[ $container_type = "test" ]]; then
    echo "Building test container...."
    dockerfile=Dockerfile.test
    docker build --tag $ecr_url/$ecr_alias/$container_name:test \
                --file $dockerfile \
                --build-arg TRANSFORMERS_VERSION=$transformers_version \
                --build-arg  DATASETS_VERSION=$datasets_version \
                . 
    exit 1
else
    echo "Pass --type cpu or --type gpu to build a container, for testing pass --type test"
    exit 1
fi

# build docker container
docker build --tag $ecr_url/$ecr_alias/$container_name:$tag \
                --file $dockerfile \
                --build-arg TRANSFORMERS_VERSION=$transformers_version \
                --build-arg  DATASETS_VERSION=$datasets_version \
                . 

# login into public ecr registry
echo "login into ecr-public registry"
aws ecr-public get-login-password \
    --region us-east-1 \
    --profile $aws_profile \
| docker login \
    --username AWS \
    --password-stdin $ecr_url/$ecr_alias

# push docker to registry
echo "pushing build docker image"
docker push $ecr_url/$ecr_alias/$container_name:$tag

