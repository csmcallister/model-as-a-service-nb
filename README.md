# model-as-a-service-nb

Testing a Notebook for Model-as-a-Service

## Build the Training Image

```bash
docker build -t custom-training-container .
```

>To train a model using the image on SageMaker, [push the image to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html) and start a SageMaker training job with the image URI passed to the `image_name` kwarg.

```python
import os

import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

bucket = os.getenv('BUCKET_NAME')
sagemaker_session = sagemaker.Session(default_bucket=bucket)
role = get_execution_role(sagemaker_session)

estimator = TensorFlow(
    image_name="custom-training-container",
    model_dir=f"s3://{bucket}",
    output_path=f"s3://{bucket}",
    role=role,
    train_instance_count=2,
    train_instance_type='ml.p2.xlarge',
    distributions={'parameter_server': {'enabled': True}})

estimator.fit()
```

## Push the Training Image

### Create an AWS ECR Repository

In this step, we'll use the AWS CDK to create an ECR repository where we can push our Docker image.

```bash
cdk deploy --profile model-as-a-service
```

When this is finished, it'll output the uri of your ECR repository.

```bash
 ✅  ModelStackRepository

Outputs:
ModelStackRepository.Output = REPO_URI
...
...
```

Take note of the `REPO_URI`.

### Push the Docker Image

Here we'll push our already-built docker image to the ECR repository we just created.

To push a Docker or Open Container Initiative (OCI) image to an Amazon ECR repository

1. Install version 2 of the AWS CLI following the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html). Make sure you remove version 1 first.

2. Authenticate your Docker client to the Amazon ECR registry to which you intend to push your image. Authentication tokens must be obtained for each registry used, and the tokens are valid for 12 hours. For more information, see [Registry Authentication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth).

```bash
aws ecr get-login-password --region us-east-1 --profile model-as-a-service | docker login --username AWS --password-stdin REPO_URI
```

3. Identify the image to push. Run the docker images command to list the images on your system.

```bash
docker images
```

>You can identify an image with the repository:tag value or the image ID in the resulting command output.

4. Tag your image with the Amazon ECR registry, repository, and optional image tag name combination to use. The registry format is aws_account_id.dkr.ecr.region.amazonaws.com. The repository name should match the repository that you created for your image. If you omit the image tag, we assume that the tag is latest.

The following example tags an image with the ID e9ae3c220b23:

```bash
docker tag e9ae3c220b23 REPO_URI
```

5. Push the image using the docker push command:

```bash
docker push REPO_URI
```
