# The name of our algorithm
docker_name=llava-next

# cd container

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=us-west-2


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${docker_name}:latest"
# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${docker_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${docker_name}" > /dev/null
fi

$(aws ecr get-login --registry-ids ${account} --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)

# Get the login command from ECR and execute it directly
docker login --username AWS --password $(aws ecr get-login-password --region ${region} ) ${account}.dkr.ecr.${region}.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# docker build  -t ${docker_name} --no-cache --progress plain . --build-arg REGION=${region} --file Dockerfile 
docker build  -t ${docker_name} --progress plain . --build-arg REGION=${region} --file Dockerfile 
docker tag ${docker_name} ${fullname}
docker push ${fullname}


