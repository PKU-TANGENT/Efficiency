import torch
import torch.nn as nn
import importlib
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from typing import List, Optional, Tuple, Union
from .swam import SWAM
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
class SWAMRobertaClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""
    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class SWAMRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.model_args = kwargs.pop('model_args', None) 
        if self.model_args is not None and self.model_args.add_prompt:
            self.soft_prompt = nn.Embedding(self.model_args.prompt_length, config.hidden_size)
            self.register_buffer(
                "prompt_range",
                torch.arange(0,self.model_args.prompt_length, dtype=torch.long).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "prompt_attention_mask",
                torch.ones(1,self.model_args.prompt_length,dtype=torch.long),
                persistent=False
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.swam = SWAM(config)
        self.classifier = SWAMRobertaClassificationHead(config)
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
        force_swam_weight: Optional[torch.FloatTensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device
        if self.model_args.add_prompt:
            batch_size = input_ids.size(0)
            batch_prompt_range = self.prompt_range.expand(batch_size,-1)
            soft_prompt = self.soft_prompt(batch_prompt_range)
            soft_prompt = self.dropout(soft_prompt)
            attention_mask = torch.concat(
                [
                    self.prompt_attention_mask.expand(batch_size, -1),
                    attention_mask
                ],
                dim=-1
            )
            new_inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)
            inputs_embeds = torch.concat(
                [
                    soft_prompt,
                    new_inputs_embeds
                ],
                dim=-2
            )
            input_ids = None


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states
        word_embedding = hidden_states[0]
        last_hidden_state = outputs.last_hidden_state
        swam_outputs, weights = self.swam(
            word_embedding=word_embedding, 
            last_hidden_state=last_hidden_state, 
            attention_mask=attention_mask,
            force_swam_weight=force_swam_weight
            )
        logits = self.classifier(swam_outputs)
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
            raise ValueError("please implement the following block")
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=weights if output_attentions else None,
        )