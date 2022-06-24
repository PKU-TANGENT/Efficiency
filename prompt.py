# %%
import os
import copy
import re
from platform import architecture
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from models.modeling_swam_roberta_toy import SWAMRobertaForSequenceClassification
from transformers import AutoTokenizer
device=torch.device("cuda")
prompt_length = 5
model_name_or_path = "fine-tune/swam-prompt-freeze-roberta-base/rte"
model = SWAMRobertaForSequenceClassification.from_pretrained(model_name_or_path, prompt_length=prompt_length).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
architecture = re.findall("SWAM(.*)ForSequenceClassification",model.config.architectures[0])[0].lower()
# %%
from datasets import load_dataset
task_name = model_name_or_path.split("/")[-1]
raw_datasets = load_dataset("glue",task_name)
# %%
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task_name]
is_regression = task_name == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1
padding="max_length"
max_seq_length = 128
# %%
def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    return result
raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
validation_dataset = raw_datasets["validation_matched" if task_name == "mnli" else "validation"]
# predict_dataset = raw_datasets["test_matched" if task_name == "mnli" else "test"]
# %%
model.eval()
batch_size = 64
input_features = ["input_ids", "attention_mask"]
correct_ids = []
incorrect_ids = []
for i in range(0, len(validation_dataset), batch_size):
    tmp_end = min(i+batch_size, len(validation_dataset))
    batch = validation_dataset[i:tmp_end]
    inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in input_features}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_labels = logits.argmax(dim=-1)
        for j in range(pred_labels.size(0)):
            if pred_labels[j].cpu().item() == batch["label"][j]:
                correct_ids.append(i+j)
            else:
                incorrect_ids.append(i+j)
# %%
import matplotlib.pyplot as plt
import random
display_batch_size = 5
def generate_display_batch():
    tmp_count = 0
    while tmp_count < len(incorrect_ids):
        yield [incorrect_ids[i + tmp_count] for i in range(display_batch_size) if i + tmp_count < len(incorrect_ids)]
        tmp_count += display_batch_size
display_batch_generator = generate_display_batch()
# %%
to_predict = next(display_batch_generator)
to_predict_dataset = validation_dataset[to_predict]
input_features = ["input_ids", "attention_mask"]
inputs = {k: torch.tensor(to_predict_dataset[k]).to(device) for k in input_features}
outputs = model(**inputs, output_attentions=True)
# %%
def get_unique_words(list_of_tokens):
    list_of_tokens = ["<sp>"] * prompt_length + list_of_tokens
    new_list_of_tokens = copy.deepcopy(list_of_tokens)
    for i in range(len(list_of_tokens)):
        if list_of_tokens[i] in list_of_tokens[:i]:
            new_list_of_tokens[i] = list_of_tokens[i]+str(list_of_tokens[:i].count(list_of_tokens[i]))
    return new_list_of_tokens
# %%
for ii in range(len(to_predict)):
    print(to_predict_dataset[sentence1_key][ii])
    print(to_predict_dataset[sentence2_key][ii])
    print("predicted: " + model.config.id2label[outputs.logits[ii].argmax().item()])
    print("should be: " + model.config.id2label[to_predict_dataset["label"][ii]])
    plt.figure(figsize = (8,6),dpi=300)
    tmp_sent_length = (inputs["input_ids"][ii]==tokenizer.pad_token_id).to(torch.int).argmax()
    tmp_sent_length = len(inputs["input_ids"][ii]) if not tmp_sent_length else tmp_sent_length
    tmp_input_ids = inputs["input_ids"][ii][:tmp_sent_length]
    # tmp_input_ids = ["<sp>"] * prompt_length + tmp_input_ids.cpu().numpy().tolist()
    tmp_output_attentions = outputs.attentions[ii][:tmp_sent_length+prompt_length]
    plt.bar(get_unique_words(tokenizer.convert_ids_to_tokens(tmp_input_ids)),tmp_output_attentions.cpu().detach().numpy())
    plt.bar(get_unique_words(tokenizer.convert_ids_to_tokens(tmp_input_ids)),(tmp_output_attentions.where(tmp_output_attentions>1/(torch.sum(inputs["attention_mask"][ii])+prompt_length),torch.zeros_like(tmp_output_attentions))).cpu().detach().numpy())
    plt.ylabel("SWAM Weights")
    plt.xlabel("Sentence Tokens")
    plt.title("Visualizing Self-Weighted Outputs")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
# %%
learned_soft_embedding = model.soft_prompt.weight
word_embedding = eval("model."+architecture+".embeddings.word_embeddings.weight")
for i in range(prompt_length):
    soft_words = (learned_soft_embedding[i,:].unsqueeze(0) * word_embedding).sum(-1).topk(5).indices
    print(tokenizer.convert_ids_to_tokens(soft_words))


# %%
ii=3
print(to_predict_dataset[sentence1_key][ii])
print(to_predict_dataset[sentence2_key][ii])
print("predicted: " + model.config.id2label[outputs.logits[ii].argmax().item()])
print("should be: " + model.config.id2label[to_predict_dataset["label"][ii]])
tmp_sent_length = (inputs["input_ids"][ii]==tokenizer.pad_token_id).to(torch.int).argmax()
tmp_input_ids = inputs["input_ids"][ii][:tmp_sent_length]
tmp_output_attentions = outputs.attentions[ii][:tmp_sent_length]
plt.figure(figsize = (8,6),dpi=300)
plt.bar(get_unique_words(tokenizer.convert_ids_to_tokens(tmp_input_ids)),tmp_output_attentions.cpu().detach().numpy())
plt.bar(get_unique_words(tokenizer.convert_ids_to_tokens(tmp_input_ids)),(tmp_output_attentions.where(tmp_output_attentions>1/torch.sum(inputs["attention_mask"][ii]),torch.zeros_like(tmp_output_attentions))).cpu().detach().numpy())
plt.ylabel("SWAM Weights")
plt.xlabel("Sentence Tokens")
plt.title("Visualizing SWAM Outputs")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
special_seperate_token = tokenizer.convert_ids_to_tokens(8820)[0]
to_modify_dataset = validation_dataset[to_predict[ii]]
input_features = ["input_ids", "attention_mask"]
to_modify_inputs = {k: torch.tensor(to_modify_dataset[k]).to(device)[None,:] for k in input_features}
to_modify_tokens = tokenizer.convert_ids_to_tokens(to_modify_inputs["input_ids"][0])
to_modify_outputs = model(**to_modify_inputs, output_attentions=True)
# %%
loss_func = torch.nn.CrossEntropyLoss()
model.zero_grad()
to_modify_outputs.attentions.retain_grad()
to_modify_logits = to_modify_outputs.logits
to_modify_pred_labels = torch.tensor([1 - to_modify_logits.argmax(dim=-1)]).to(device)
loss = loss_func(to_modify_logits, to_modify_pred_labels)
loss.backward(retain_graph=True)
# print(to_modify_outputs.attentions.grad)
#%%
plt.figure(figsize = (8,6),dpi=300)
plt.bar(get_unique_words(tokenizer.convert_ids_to_tokens(tmp_input_ids)),to_modify_outputs.attentions.grad[0, :tmp_sent_length].cpu().detach().numpy())
plt.ylabel("Back-Prop Gradients")
plt.xlabel("Sentence Tokens")
plt.title("Visualizing Back-Prop Gradients")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#%%
to_change_index = to_modify_tokens.index(special_seperate_token+input())
print(to_change_index)
print(to_modify_outputs.attentions[0][to_change_index])
print(to_modify_outputs.attentions.grad[0][to_change_index])
# %%
modified_weights = to_modify_outputs.attentions
modified_weights[0][to_change_index] = eval(input())
modified_output = model(**to_modify_inputs, output_attentions=True, force_swam_weight=modified_weights)
print(modified_output.logits)
# %%
