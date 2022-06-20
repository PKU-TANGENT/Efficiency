# %%
import os
from xml.etree.ElementPath import prepare_descendant
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from models.modeling_swam_roberta import SWAMRobertaForSequenceClassification
from transformers import AutoTokenizer
device=torch.device("cuda")
model_name_or_path = "/home/zhejian/emnlp/cls/fine-tune/swam-roberta-base/rte"
model = SWAMRobertaForSequenceClassification.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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
total_success = 0
loss_func = torch.nn.CrossEntropyLoss()
for i in range(len(incorrect_ids)):
    tmp_input = validation_dataset[incorrect_ids[i]]
    prepared_input = {k: torch.tensor(tmp_input[k]).to(device)[None,:] for k in input_features}
    tmp_sent_length = torch.tensor(prepared_input["input_ids"]==tokenizer.pad_token_id).to(torch.int).argmax()
    tmp_sent_length = prepared_input["input_ids"].size(-1) if tmp_sent_length == 0 else tmp_sent_length
    to_modify_outputs = model(**prepared_input, output_attentions=True)
    to_modify_outputs.attentions.retain_grad()
    model.zero_grad()
    to_modify_logits = to_modify_outputs.logits
    to_modify_pred_labels = torch.tensor([1 - to_modify_logits.argmax(dim=-1)]).to(device)
    loss = loss_func(to_modify_logits, to_modify_pred_labels)
    loss.backward(retain_graph=True)
    tmp_grad = to_modify_outputs.attentions.grad[0, :tmp_sent_length]
    to_modify_index = tmp_grad.abs().argmax()
    modified_weights = to_modify_outputs.attentions 
    modified_weights[0, to_modify_index] = -10 if tmp_grad[to_modify_index] > 0 else 10
    modified_output = model(**prepared_input, force_swam_weight=modified_weights)
    total_success += (modified_output.logits.argmax(dim=-1) == tmp_input["label"]).item()
print(f"Success edit rate: {total_success/len(incorrect_ids)}")