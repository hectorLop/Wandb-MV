
<p align="center">
  <img src="docs/wandb_mv_logo.png" alt="wandb_mv_logo" width=550 />
</p>

### Description
This library provides an abstraction to perform Model Versioning using Weight & Biases.

### Installation
The package can be installed using pip:

```bash
pip install wandb-mv
```

### Features
- Version a new trained model
- Promote a model to another stage (e.g production)

### Example of usage

The following code snippet shows how to promote a newly trained model to the `best_model` on the validation set. First, it creates the model providing the checkpoint path, the artifact name and type, a description, and the metadata, where the metrics or any desired information is stored.

The `comparision_type` indicates how to compare the metrics between the two models. E.g if `smaller`, the smaller the metric from the new model the better

```python
run = wandb.init(...)

# Create the desired artifact
artifact = versioner.create_artifact(
                            checkpoint='model.ckpt',
                            artifact_name='prueba',
                            artifact_type='model',
                            description='Prueba Wandb-MV',
                            metadata={
                                'val_metric': 78.0,
                                'test_metric': 0.0
                            })

# Promote the desired artifact to the 'best_model' tag
versioner.promote_model(new_model=artifact,
                        artifact_name='prueba',
                        artifact_type='model',
                        comparision_metric='val_metric',
                        promotion_alias='best_model',
                        comparision_type='smaller'
                       )
```

This code snippet shows how to promote a trained model to production after being validated on the test set. The `already_deployed` parameter indicates that the model is already logged, so it only needs to be updated.

```python
versioner = Versioner(run)
versioner.promote_model(new_model=artifact,
                        artifact_name='prueba',
                        artifact_type='model',
                        comparision_metric='test_metric',
                        promotion_alias='production',
                        comparision_type='smaller',
                        already_deployed=True
                        )
```

### Next features
- [ ] Allow providing a custom comparision function to promote a model
- [ ] ...



