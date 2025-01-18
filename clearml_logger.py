from pytorch_lightning.loggers import Logger as LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Any, Dict, Optional, Union
import torch

class ClearMLLogger(LightningLoggerBase):
    def __init__(self, task, upload_interval=100):
        super().__init__()
        self.task = task
        self._logger = task.get_logger()
        self.upload_interval = upload_interval
        self.best_val_loss = float('inf')

    @property
    def name(self) -> str:
        return "ClearMLLogger"

    @property
    def version(self) -> Union[int, str]:
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        # ClearML automatically logs hyperparameters through task.connect()
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            title, series = k.split('/') if '/' in k else (k, k)
            self._logger.report_scalar(title=title, series=series, value=v, iteration=step)
        
        # Handle model upload based on validation loss
        if 'loss/val' in metrics:
            val_loss = metrics['loss/val']
            epoch = step if step is not None else 0
            
            if val_loss < self.best_val_loss or epoch % self.upload_interval == 0:
                self.best_val_loss = min(val_loss, self.best_val_loss)
                self._upload_model(epoch, metrics)

    def _upload_model(self, epoch, metrics):
        model_filename = f'model_epoch_{epoch}_valloss_{metrics["loss/val"]:.4f}.pt'
        
        # Get the model from the trainer
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'model'):
            torch.save(self.trainer.model.state_dict(), model_filename)
            
            self.task.upload_artifact(
                name=f'model_epoch_{epoch}',
                artifact_object=model_filename,
                metadata={
                    'epoch': epoch,
                    'val_loss': float(metrics.get('loss/val', 0)),
                    'train_loss': float(metrics.get('loss/train', 0)),
                }
            )

    @property
    def save_dir(self) -> Optional[str]:
        return "."

    @rank_zero_only
    def finalize(self, status: str) -> None:
        pass
