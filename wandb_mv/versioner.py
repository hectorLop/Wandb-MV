from typing import Dict, List, Optional
import wandb
from wandb_mv.utils import COMP_FUNC
from wandb.wandb_run import Run

class Versioner():
    """
    Weight & Biases Model Versioning

    Args:
        run (wandb.wandb_run.Run): Wandb experiment
    """

    def __init__(self, run : Run) -> None:
        self.run = run

    def create_artifact(
        self,
        artifact_name : str,
        artifact_type : str,
        description : str,
        checkpoint : str = '',
        aliases : Optional[List[str]] = None,
        metadata : Optional[Dict] = None,
        publish : bool = False
    ) -> wandb.Artifact:
        """
        Creates a new artifact

        Args:
            checkpoint (str): Checkpoint location.
            artifact_name (str): Artifact's desired name.
            artifact_type (str): Type of the artifact.
            description (str): Description of the new artifact.
            aliases (Optional[List[str]]): Aliases to add to the artifact.
                Default is None, thus it adds the 'latest' alias by default.
            metadata (Optional[Dict]): Artifact's metadata. Default is None.
            publish (bool): Flag to publish the model in the current experiment.
                Default is False
        
        Returns:
            wandb.Artifact: Created artifact.
        """
        artifact = wandb.Artifact(name=artifact_name,
                                  type=artifact_type,
                                  description=description,
                                  metadata=metadata)

        if checkpoint:
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
        comparision_type : str = 'smaller',
        already_deployed : bool = False
    ) -> None:
        """
        Promote a model based on a given criterion.

        Args:
            new_model (wandb.Artifact): New model's artifact.
            artifact_name (str): Artifact's desired name.
            artifact_type (str): Type of the artifact.
            comparision_metric (str): Key of the metadata to obtain the 
                metric that will decide if promote the new model.
            promotion_alias (str): Alias that defines the stage where
                to promote the model.
            comparision_type (str): Type of comparision. Default is smaller,
                so the new model will be promoted if the promoted model's 
                metric is smaller than the new one. The comparision types are:
                    - smaller
                    - smaller_or_equal 
                    - greater
                    - greater_or_equal
            already_deployed (bool): Flag to decide if the new model is 
                already deployed, so it only need to change the aliases.
        """
        # Get the promoted model if exists
        try:
            promoted_model = self.run.use_artifact(
                                        f'{artifact_name}:{promotion_alias}',
                                        type=artifact_type)
        except:
            promoted_model = None
        
        # TODO: Check if by default the 'latest' alias is added even though
        # other aliases are appended
        aliases = ['latest']
        
        if promoted_model:
            # Get the promoted model and the new model metrics
            promoted_model_metric = promoted_model.metadata[comparision_metric]
            new_model_metric = new_model.metadata[comparision_metric]

            # Get the comparision function
            compare_func = COMP_FUNC[comparision_type]

            if compare_func(promoted_model_metric, new_model_metric):  
                # If the model is already deployed, we only need to persist
                # the changes              
                if already_deployed:
                    new_model.aliases.append(promotion_alias)
                    new_model.save()
                else:
                    aliases.append(promotion_alias)

                # Persist changes on the promoted model
                promoted_model.aliases.remove(promotion_alias)
                promoted_model.save()

                print('Promoted new model')
            else:
                print('This new model does not improve the older one')
        else: 
            aliases.append(promotion_alias)
            
            if already_deployed:
                new_model.aliases.append(promotion_alias)
                new_model.save()
                
            msg = (
                f'There is no artifact named {artifact_name}:{promotion_alias}',
                'so the new model is promoted'        
            )

            print(*msg)

        # If the new model wasn't deployed, it must to be logged
        if not already_deployed:
            self.run.log_artifact(new_model, aliases=aliases)

    def get_latest_version(self, name : str) -> int:
        """
        Get the latest model version.

        Args:
            name (str): Model's name

        Returns:
            int: Model's version. If it does not exist, it returns -1, so 
                the new version is the v0 version. 
        """
        try: 
            artifact = self.run.use_artifact(f'{name}:latest')
        except:
            return -1

        return int(artifact.version[1])