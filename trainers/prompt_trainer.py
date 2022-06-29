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

class PromptTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            decay_criterion = lambda n: n in decay_parameters
            head_criterion = lambda n: "classifier" in n
            prompt_criterion = lambda n: "soft_prompt" in n
            head_decay = [p for n, p in opt_model.named_parameters() if head_criterion(n) and decay_criterion(n)]
            head_not_decay = [p for n, p in opt_model.named_parameters() if head_criterion(n) and not decay_criterion(n)]
            prompt = [p for n, p in opt_model.named_parameters() if prompt_criterion(n)]
            other_decay = [p for n,p in opt_model.named_parameters() if not head_criterion(n) and not prompt_criterion(n) and decay_criterion(n)]
            other_not_decay = [p for n,p in opt_model.named_parameters() if not head_criterion(n) and not prompt_criterion(n) and not decay_criterion(n)]
            optimizer_grouped_parameters = [
                {
                    "params": other_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": other_not_decay,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                {
                    "params": head_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.model.model_args.model_head_lr
                },

                {
                    "params": head_not_decay,
                    "weight_decay": 0.0,
                    "lr": self.model.model_args.model_head_lr
                },
                {
                    "params": prompt,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.model.model_args.prompt_lr
                },
            ]

            optimizer_cls, optimizer_kwargs = PromptTrainer.get_optimizer_cls_and_kwargs(self.args)
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