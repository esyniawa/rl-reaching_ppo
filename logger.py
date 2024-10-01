from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger, BasicLogger
from tianshou.utils.logger.base import LOG_DATA_TYPE


class CustomLogger(BasicLogger):
    def __init__(self, writer: SummaryWriter, log_path: str):
        super().__init__(self.write_log)
        self.writer = writer
        self.log_path = log_path
        self.last_log = {}

    def write_log(self, data: LOG_DATA_TYPE):
        for k, v in data.items():
            self.writer.add_scalar(k, v, data['env_step'])

        # Print to console
        log_str = f"Epoch #{data.get('epoch', 0)}: "
        log_str += f"env_step={data.get('env_step', 0)}, "
        log_str += f"rew={data.get('rew', 0):.2f}, "
        log_str += f"len={data.get('len', 0):.1f}, "
        log_str += f"loss={data.get('loss', 0):.3f}"
        print(log_str)

        self.last_log = data

    def save_data(self, epoch: int, env_step: int, gradient_step: int) -> None:
        pass  # We don't need this for now
