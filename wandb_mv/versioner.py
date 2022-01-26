from typing import Dict, List, Optional
import wandb
from wandb_mv.utils import COMP_FUNC
from wandb.wandb_run import Run

class Versioner():

    def __init__(self, run : Run) -> None:
        self.run = run

    def create_artifact(
        self,
        checkpoint : str,
        artifact_name : str,
        artifact_type : str,
        description : str,
        aliases : Optional[List[str]] = None,
        metadata : Optional[Dict] = None,
        publish : bool = False
    ) -> None:
        artifact = wandb.Artifact(name=artifact_name,
                                  type=artifact_type,
                                  description=description,
                                  metadata=metadata)
        artifact.add_file(checkpoint)

        if publish:
            self.run.log_artifact(artifact, aliases=aliases)
        
        return artifact

    def promote_model(
        self,
        new_model : wandb.Artifact,
        artifact_name : str,
        artifact_type : str,
        comparision_metric : str,
        promotion_alias : str,
        comparision_type : str
    ) -> None:
        try:
            promoted_model = self.run.use_artifact(
                                        f'{artifact_name}:{promotion_alias}',
                                        type=artifact_type)
        except:
            promoted_model = None
        
        aliases = ['latest']
        
        if promoted_model:
            promoted_model_metric = promoted_model.metadata[comparision_metric]
            new_model_metric = new_model.metadata['val_metric']
            compare_func = COMP_FUNC[comparision_type]

            if compare_func(promoted_model_metric, new_model_metric):
                aliases.append(promotion_alias)

                promoted_model.aliases.remove(promotion_alias)
                promoted_model.save()

                print('Promoted new model')
            else:
                print('This new model does not improve the older one')
        else:
            aliases.append(promotion_alias)
            msg = (
                f'There is no artifact named {artifact_name}:{promotion_alias}',
                'so the new model is promoted'
            
            )
            print(*msg)
            
        self.run.log_artifact(new_model)