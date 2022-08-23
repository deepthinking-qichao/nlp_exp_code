from transformers import BertTokenizer, BertForSequenceClassification
from dataset import MyDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from func import Ts_loss_cal

model_path = 'C:/workspace/code/BERT_model/chinese-bert-wwm'
data_path = 'C:/workspace/code/Dataset/new_waimai/'

batch_size = 2
shuffle = False
num_workers = 8
num_epochs = 3
ts_flag = True

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

if ts_flag:
    teacher_model_path = 'C:/workspace/code/BERT_model/chinese-roberta-wwm-ext-large'
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_path, num_labels=2)
    ts_loss_cal = Ts_loss_cal()

train_dastset_path = data_path + "waimai_10k_train.csv"
train_dataset = MyDataset(tokenizer, train_dastset_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=train_dataset.collate_fn)
test_dastset_path = data_path + "waimai_10k_test.csv"
test_dataset = MyDataset(tokenizer, test_dastset_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=test_dataset.collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

if ts_flag:
    teacher_model.to(device)
    teacher_model.eval()

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_texts = batch.input_texts.to(device)
        labels = torch.tensor(batch.label).to(device)
        reviews = batch.review

        outputs = model(
            input_ids = input_texts["input_ids"], attention_mask=input_texts["attention_mask"], token_type_ids=input_texts["token_type_ids"], labels=labels)
        
        ts_loss = 0
        if ts_flag:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                input_texts["input_ids"], attention_mask=input_texts["attention_mask"], token_type_ids=input_texts["token_type_ids"], labels=labels)
                ts_loss += ts_loss_cal(teacher_outputs.logits, outputs.logits)

        loss = outputs.loss
        loss += ts_loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    prediction_all = []
    label_all = []
    for batch in test_dataloader:
        input_texts = batch.input_texts.to(device)
        labels = torch.tensor(batch.label).to(device)
        
        reviews = batch.review
        with torch.no_grad():
            outputs = model(
            input_texts["input_ids"], attention_mask=input_texts["attention_mask"], token_type_ids=input_texts["token_type_ids"])

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        label_all.append(labels.cpu())
        prediction_all.append(predictions.cpu())
    label_all = torch.cat(label_all, dim=0).numpy()
    prediction_all = torch.cat(prediction_all, dim=0).numpy()
    print(classification_report(label_all, prediction_all))

    print("yes")

    print("no")



