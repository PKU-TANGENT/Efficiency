import torch
import torch.nn as nn
from .lora import Lora
from transformers.models.roberta.modeling_roberta import (
    RobertaModel, 
    RobertaEncoder, 
    RobertaPooler,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaForSequenceClassification,
    RobertaAttention,
    RobertaSelfAttention,
    RobertaSelfOutput,
    RobertaOutput,
    RobertaIntermediate
)
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import SequenceClassifierOutput
from .modeling_utils import PoolerClassificationHead
class LoraRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, **kwargs):
        self.model_args = kwargs.pop('model_args', None)
        config.lora_rank = self.model_args.lora_rank if self.model_args is not None else 1
        config.elementwise_affine = self.model_args.elementwise_affine if self.model_args is not None else True
        config.lora_layers=list(map(int,self.model_args.lora_layers.split(","))) if self.model_args is not None else [10] 
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = LoraRobertaModel(config, add_pooling_layer=False)
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


class LoraRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.encoder = LoraRobertaEncoder(config)
        self.embeddings = RobertaEmbeddings(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
class LoraRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()
        self.config = config
        assert config.num_hidden_layers > 1
        tmp_layer_list=[]
        for i in range(config.num_hidden_layers):
            if i in config.lora_layers:
                tmp_layer_list.append(LoraRobertaLayer(config))
            else:
                tmp_layer_list.append(RobertaLayer(config))


        self.layer = nn.ModuleList(tmp_layer_list)
        self.gradient_checkpointing = False

class LoraRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super(RobertaLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LoraRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = LoraRobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

class LoraRobertaAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super(RobertaAttention, self).__init__()
        self.self = LoraRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

class LoraRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super(RobertaSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = Lora(config.hidden_size, self.all_head_size, config.lora_rank)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = Lora(config.hidden_size, self.all_head_size, config.lora_rank)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder


    
    
