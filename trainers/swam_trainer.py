from transformers import Trainer
import numpy as np
import torch
import torch.nn as nn
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.integrations import is_fairscale_available
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    )
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    ShardedDDPOption,
    denumpify_detensorize,
    has_length,
)
if is_fairscale_available():
    from fairscale.optim import OSS
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
logger = logging.get_logger(__name__)

class SWAMTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            tmp_criterion = lambda x: "classifier" in x or "swam" in x
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and not tmp_criterion(n)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and tmp_criterion(n)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.model.model_args.model_head_lr
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and not tmp_criterion(n)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and tmp_criterion(n)],
                    "weight_decay": 0.0,
                    "lr": self.model.model_args.model_head_lr
                },

            ]

            optimizer_cls, optimizer_kwargs = SWAMTrainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr", None)
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer