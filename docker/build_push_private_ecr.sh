#!/bin/bash

# huggingface versions defaults to latest
transformers_version=$(curl --silent --location https://pypi.org/pypi/transformers/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1].strip())")
datasets_version=$(curl --silent --location https://pypi.org/pypi/datasets/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1].strip())")



# image variables
image_type=training
container_name=huggingface-$image_type
device=cpu
                      
# container versions/tags
version=0.0.1
cpu_tag=$version-cpu-transformers$transformers_version-datasets$datasets_version
gpu_tag=$version-gpu-transformers$transformers_version-datasets$datasets_version-cu110

# aws parameters
aws_profile=hf-sm
aws_account_id=$(aws sts get-caller-identity --profile $aws_profile | python -c "import sys, json; aws_account = json.load(sys.stdin)['Account']; print(aws_account)")
aws_region=eu-central-1

#registry parameters
ecr_url=$aws_account_id.dkr.ecr.$aws_region.amazonaws.com

# parsing named arguments
while (( $# > 1 )); do case $1 in
   --device) device="$2";;
   --image_type) image_type="$2";;
   --profile) aws_profile="$2";;
   --transformers_version) transformers_version="$2";;
   --datasets_version) datasets_version="$2";;
   --version) version="$2";;
   *) break;
 esac; shift 2
done

# can be run only once! public repositories has to be created in us-east-1
aws ecr create-repository --repository-name $container_name  --profile $aws_profile --region $aws_region   > /dev/null

# Starting print 
echo "####################### Build and Push HuggingFace Sagemaker Container #######################"
echo ""
echo "Build parameters:"
echo ""

echo "      image_type=$(tput setaf 2)${image_type}$(tput sgr0)"
echo "      device=$(tput setaf 2)${device}$(tput sgr0)"
echo "      account_id=$(tput setaf 2)${aws_account_id}$(tput sgr0)"
echo "      profile=$(tput setaf 2)${aws_profile}$(tput sgr0)"
echo "      transformers_version=$(tput setaf 2)${transformers_version}$(tput sgr0)"
echo "      datasets_version=$(tput setaf 2)${datasets_version}$(tput sgr0)"
echo "      container_version=$(tput setaf 2)${version}$(tput sgr0)"
echo ""

# checks if $aws_profile is available
if [[ $aws_profile != "ci" ]]; then
    profile_status=$( (aws configure --profile ${aws_profile} list ) 2>&1 )
    if [[ $profile_status = *'could not be found'* ]]; then 
        echo "" 
        echo "$(tput setaf 1)Failure: AWS proifle ${aws_profile} not found$(tput sgr0)" 
        exit 1
    fi
fi
# extracts container type  
if [[ $device = "gpu" ]]; then
    echo "Building gpu container...."
    tag=$gpu_tag
    dockerfile=Dockerfile.gpu
elif [[ $device = "cpu" ]]; then
    echo "Building cpu container...."
    tag=$cpu_tag
    dockerfile=Dockerfile.cpu
elif [[ $device = "test" ]]; then
    echo "Building test container...."
    dockerfile=Dockerfile.test
    docker build --tag $ecr_url/$ecr_alias/$container_name \
                --file ./$image_type/$dockerfile \
                --build-arg TRANSFORMERS_VERSION=$transformers_version \
                --build-arg  DATASETS_VERSION=$datasets_version \
                 ./$image_type 
    exit 1
else
    echo "" 
    echo "$(tput setaf 1)Failure: Pass --type cpu or --type gpu to build a container, for testing pass --type test $(tput sgr0)" 
    exit 1
fi

# build docker container
docker build --tag $ecr_url/$container_name:$tag \
                --file ./$image_type/$dockerfile \
                --build-arg TRANSFORMERS_VERSION=$transformers_version \
                --build-arg  DATASETS_VERSION=$datasets_version \
                ./$image_type 

# login into public ecr registry
echo "login into ecr-public registry using ${aws_profile} profile"
if [[ $aws_profile = 'ci' ]]; then
    aws ecr get-login-password \
        --region $aws_region \
    | docker login \
        --username AWS \
        --password-stdin $ecr_url
else
    aws ecr get-login-password \
        --region $aws_region \
        --profile $aws_profile \
    | docker login \
        --username AWS \
        --password-stdin $ecr_url
fi

# push docker to registry
echo "pushing build docker image"
docker push $ecr_url/$container_name:$tag