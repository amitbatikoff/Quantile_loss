from pytorch_lightning.loggers import Logger as LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Any, Dict, Optional, Union

class ClearMLLogger(LightningLoggerBase):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self._logger = task.get_logger()

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
            # Use metric name as both title and series to organize metrics better
            title, series = k.split('/') if '/' in k else (k, k)
            self._logger.report_scalar(title=title, series=series, value=v, iteration=step)

    @property
    def save_dir(self) -> Optional[str]:
        return "."

    @rank_zero_only
    def finalize(self, status: str) -> None:
        pass
