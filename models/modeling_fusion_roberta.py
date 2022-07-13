import torch
import math
import torch.nn as nn
from .adapter import Adapter
from .lora import Lora
from transformers.models.roberta.modeling_roberta import (
    RobertaModel, 
    RobertaEncoder, 
    RobertaPooler,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaForSequenceClassification,
    RobertaSelfOutput
)
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modeling_utils import PoolerClassificationHead
def modify_lora_layer(layer, config):
    targets = config.lora_target
    if "q" in targets:
        layer.attention.self.query = Lora(
            config.hidden_size, 
            layer.attention.self.all_head_size, 
            config.lora_rank
            ) if not config.rand_init else LoraRandInit(
            config.hidden_size, 
            layer.attention.self.all_head_size, 
            config.lora_rank)
    if "k" in targets:
        layer.attention.self.key = Lora(
            config.hidden_size, 
            layer.attention.self.all_head_size, 
            config.lora_rank
            ) if not config.rand_init else LoraRandInit(
            config.hidden_size, 
            layer.attention.self.all_head_size, 
            config.lora_rank)
    if "v" in targets:
        layer.attention.self.value = Lora(config.hidden_size, layer.attention.self.all_head_size, config.lora_rank)
    if "d" in targets:
        layer.attention.output.dense = Lora(config.hidden_size, config.hidden_size, config.lora_rank)
    return layer
class FusionRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, **kwargs):
        self.model_args = kwargs.pop('model_args', None)
        
        config.is_parallel = self.model_args.is_parallel if self.model_args is not None else False
        config.project_dim = self.model_args.project_dim if self.model_args is not None else 1
        config.adapter_layers=list(map(int,self.model_args.adapter_layers.split(","))) if self.model_args is not None else [5] 
        config.position=self.model_args.position if self.model_args is not None else "ffn"
        config.elementwise_affine = self.model_args.elementwise_affine if self.model_args is not None else True
        config.identity_init = self.model_args.identity_init if self.model_args is not None else False
        config.lora_rank = self.model_args.lora_rank if self.model_args is not None else 1
        config.lora_layers=list(map(int,self.model_args.lora_layers.split(","))) if self.model_args is not None else [10] 
        config.lora_target=list(self.model_args.lora_target.split(",")) if self.model_args is not None else ["q","k"]
        config.rand_init = self.model_args.rand_init if self.model_args is not None else False
        config.prompt_length = self.model_args.prompt_length if self.model_args is not None else 2
        config.prompt_layers=list(map(int,self.model_args.prompt_layers.split(","))) if self.model_args is not None else [-1] 

        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = FusionRobertaModel(config, add_pooling_layer=False)
        pooler_type = self.model_args.pooler_type if self.model_args is not None else "avg"
        self.classifier = PoolerClassificationHead(config, pooler_type=pooler_type)
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        to_pass_attention = None if self.classifier.pooler_type == "cls" else attention_mask
        logits = self.classifier(last_hidden_state, to_pass_attention)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FusionRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.encoder = FusionRobertaEncoder(config)
        self.embeddings = RobertaEmbeddings(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
class FusionRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()
        self.config = config
        assert config.num_hidden_layers > 1
        tmp_layer_list=[]
        for i in range(config.num_hidden_layers):
            if i in config.adapter_layers:
                tmp_layer_list.append(AdapterRobertaLayer(config))
            else:
                tmp_layer_list.append(RobertaLayer(config))
        for i in range(config.num_hidden_layers):
            if i in config.lora_layers:
                tmp_layer_list[i] = modify_lora_layer(tmp_layer_list[i], config)
        self.layer = nn.ModuleList(tmp_layer_list)
        self.gradient_checkpointing = False

class AdapterRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config) if "ffn" in config.position else None
        # self.adapter_ = Adapter(config) if "attn" in config.position else None
        if "ffn" in config.position:
            if config.is_parallel:
                self.feed_forward_chunk = self.parallel_feed_forward_chunk
            else:
                self.feed_forward_chunk = self.sequential_feed_forward_chunk
        if "attn" in config.position:
            self.attention.output = AdapterRobertaSelfOutput(config)

    def sequential_feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        adapter_output = self.adapter(layer_output)
        return adapter_output
    def parallel_feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        adapter_output = self.adapter(attention_output)
        layer_output = self.output(intermediate_output, adapter_output)
        return layer_output

class AdapterRobertaSelfOutput(RobertaSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.adapter(hidden_states)
        return hidden_states

class LoraRandInit(Lora):
    def reset_parameters(self) -> None:
        super(Lora, self).reset_parameters()
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_up, a=math.sqrt(5))