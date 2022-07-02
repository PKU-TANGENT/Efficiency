import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertEncoder,
    BertLayer,
    BertPooler,
    BertEmbeddings,
    BertForSequenceClassification
)
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    )
from .adapter import Adapter
from .modeling_utils import PoolerClassificationHead
class AdapterBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        self.model_args = kwargs.pop('model_args', None)
        config.project_dim = self.model_args.project_dim if self.model_args is not None else 1
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = AdapterBertModel(config, add_pooling_layer=False)
        pooler_type = self.model_args.pooler_type if self.model_args is not None else "cls"
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

        outputs = self.bert(
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


class AdapterBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = AdapterBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
class AdapterBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        assert config.num_hidden_layers > 1
        tmp_layer_list = [BertLayer(config) for _ in range(config.num_hidden_layers-2)]
        tmp_layer_list.append(AdapterBertLayer(config))
        tmp_layer_list.append(BertLayer(config))
        self.layer = nn.ModuleList(tmp_layer_list)
        self.gradient_checkpointing = False

class AdapterBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        adapter_output = self.adapter(layer_output)
        return adapter_output
    
    
