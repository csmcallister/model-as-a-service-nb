from aws_cdk import (
    aws_ecr as ecr,
    core,
)


# TODO: set account and region using the env property on the stack

class ModelStackRepository(core.Stack):
    def __init__(self, app: core.App, id: str) -> None:
        super().__init__(app, id)
        
        repository = ecr.Repository(self, "ModelStackRepository",
            image_scan_on_push=True,
            repository_name='sagemaker-model-repo'  # must have sagemaker in it
        )
        
        core.CfnOutput(self, "Output", value=repository.repository_uri)


app = core.App()
ModelStackRepository(app, "ModelStackRepository")
app.synth()