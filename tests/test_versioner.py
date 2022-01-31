from torch import isin
from wandb import wandb
from wandb_mv.versioner import Versioner
import os
import wandb

os.environ['WANDB_MODE'] = 'offline'

def test_create_artifact():
    run = wandb.init(project='test')

    versioner = Versioner(run)

    artifact = versioner.create_artifact(
                            artifact_name='prueba',
                            artifact_type='model',
                            description='Test Wandb-MV',
                            metadata={
                                'val_metric': 78.0,
                                'test_metric': 0.0
                            })

    assert artifact
    assert isinstance(artifact, wandb.Artifact)