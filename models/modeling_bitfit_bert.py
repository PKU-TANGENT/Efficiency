import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel, 
    BertEncoder, 
    BertPooler,
    BertLayer,
    BertEmbeddings,
    BertForSequenceClassification,
)
import re
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modeling_utils import PoolerClassificationHead
from .bitfit import BitfitLayerNorm, BitfitLinear
class BitfitBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        self.model_args = kwargs.pop('model_args', None)
        config.bitfit_selection = self.model_args.bitfit_selection.split(",") if self.model_args is not None else [
            'attention.self.query',
            'intermediate.dense', 
            ]
        config.bitfit_layers=list(map(int,self.model_args.bitfit_layers.split(","))) if self.model_args is not None else [10] 
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BitfitBertModel(config, add_pooling_layer=False)
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


class BitfitBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super(BertModel, self).__init__(config)
        self.config = config
        self.encoder = BitfitBertEncoder(config)
        self.embeddings = BertEmbeddings(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
class BitfitBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        assert config.num_hidden_layers > 1
        tmp_layer_list=[]
        for i in range(config.num_hidden_layers):
            if i in config.bitfit_layers:
                tmp_layer_list.append(BitfitBertLayer(config))
            else:
                tmp_layer_list.append(BertLayer(config))

        self.layer = nn.ModuleList(tmp_layer_list)
        self.gradient_checkpointing = False

class BitfitBertLayer(BertLayer):
    '''
    for a specific layer L
    [n for n, p in L.named_parameters() if "bias" in n] 
    yeilds:
    [
        'attention.self.query.bias', 
        'attention.self.key.bias', 
        'attention.self.value.bias', 
        'attention.output.dense.bias', 
        'attention.output.LayerNorm.bias', 
        'intermediate.dense.bias', 
        'output.dense.bias', 
        'output.LayerNorm.bias'
    ]

    '''
    def __init__(self, config):
        super().__init__(config)
        for tmp_target in config.bitfit_selection:
            tmp_target_class = str(eval("self."+tmp_target))
            part_split = tmp_target.rfind(".")
            first_part = tmp_target[:part_split]
            target_attr = tmp_target[part_split+1:]
            # tmp_target_type = re.findall("^<class 'torch\.nn\.modules\.normalization\.(\w+)'>$", tmp_target_class)[0]
            setattr(eval("self."+first_part),target_attr,eval("Bitfit"+tmp_target_class))
        # super(BertLayer, self).__init__()
        # bitfit_attention = any(["attention" in i for i in config.bitfit_selection])
        # bitfit_intermediate = any(["intermediate" in i for i in config.bitfit_selection])
        # bitfit_output = any(["output" in i for i in config.bitfit_selection])
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # self.seq_len_dim = 1
        # # self.attention = BertAttention(config) if not bitfit_attention else BitfitBertAttention(config)
        # self.attention = BertAttention(config)
        # self.is_decoder = config.is_decoder
        # self.add_cross_attention = config.add_cross_attention
        # if self.add_cross_attention:
        #     if not self.is_decoder:
        #         raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
        #     # self.crossattention = BertAttention(config, position_embedding_type="absolute") if not bitfit_attention else BitfitBertAttention(config, position_embedding_type="absolute")
        #     self.crossattention = BertAttention(config, position_embedding_type="absolute")
        # # self.intermediate = BertIntermediate(config) if not bitfit_intermediate else BitfitBertIntermediate(config)
        # self.intermediate = BertIntermediate(config)
        # # self.output = BertOutput(config) if not bitfit_output else BitfitBertOutput(config)
        # self.output = BertOutput(config)
# class BitfitBertAttention(BertAttention):
#     def __init__(self, config, position_embedding_type=None):
#         super(BertAttention, self).__init__()
#         bitfit_selfattn = any(["self" in i for i in config.bitfit_selection])
#         bitfit_output = any(["attention.output" in i for i in config.bitfit_selection])   
#         self.self = BertSelfAttention(
#             config, 
#             position_embedding_type=position_embedding_type
#             ) if not bitfit_selfattn else BitfitBertSelfAttention(
#                 config,
#                 position_embedding_type=position_embedding_type
#             )
#         self.output = BertSelfOutput(config) if not bitfit_output else BitfitBertSelfOutput(config)
#         self.pruned_heads = set()    

