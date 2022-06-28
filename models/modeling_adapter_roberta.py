import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel, 
    RobertaModel, 
    RobertaClassificationHead, 
    RobertaEncoder, 
    RobertaPooler,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaForSequenceClassification
)
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    )
class PoolerRobertaClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, pooler_type="cls"):
        super().__init__(config)
        self.pooler_type = pooler_type

    def forward(self, features, attention_mask=None):
        if self.pooler_type == "cls":
            features = features[:, 0]
        elif self.pooler_type == "avg" and attention_mask is not None:
            features = (features * attention_mask.unsqueeze(-1)).sum(axis=-2) / attention_mask.sum(axis=-1).unsqueeze(-1)
        else:
            raise NotImplementedError
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class AdapterRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, **kwargs):
        self.model_args = kwargs.pop('model_args', None)
        config.project_dim = self.model_args.project_dim if self.model_args is not None else 1
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = AdapterRobertaModel(config, add_pooling_layer=False)
        self.classifier = PoolerRobertaClassificationHead(config, pooler_type=self.model_args.pooler_type)
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


class AdapterRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.encoder = AdapterRobertaEncoder(config)
        self.embeddings = RobertaEmbeddings(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
class AdapterRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()
        self.config = config
        assert config.num_hidden_layers > 1
        tmp_layer_list = [RobertaLayer(config) for _ in range(config.num_hidden_layers-2)]
        tmp_layer_list.append(AdapterRobertaLayer(config))
        tmp_layer_list.append(RobertaLayer(config))
        self.layer = nn.ModuleList(tmp_layer_list)
        self.gradient_checkpointing = False

class AdapterRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        adapter_output = self.adapter(layer_output)
        return adapter_output


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.project_dim)
        self.activation = nn.Tanh()
        self.up_project = nn.Linear(config.project_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = self.down_project(hidden_states)
        outputs = self.activation(outputs)
        outputs = self.up_project(outputs)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(hidden_states + outputs)
        return outputs
    
    
