from transformers import Trainer
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import (
    get_parameter_names,    
)
from transformers.integrations import is_fairscale_available
from transformers.utils import (
    is_sagemaker_mp_enabled,  
    logging  
)
from transformers.trainer_utils import (
    ShardedDDPOption,
)
if is_fairscale_available():
    from fairscale.optim import OSS
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
logger = logging.get_logger(__name__)

class SelfWeightedTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            tmp_criterion = lambda x: "classifier" in x or "self_weighted" in x
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

            optimizer_cls, optimizer_kwargs = SelfWeightedTrainer.get_optimizer_cls_and_kwargs(self.args)
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
