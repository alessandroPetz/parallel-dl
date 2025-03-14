from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
import evaluate
import time



raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ELIMIARE LE COLONNE (e altro )NEL DATASET CHE NON CI INTERESSANO (La classe trainer lo fa in automatico)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
#print("colonne rimaste nel dataset:")
#print(tokenized_datasets["train"].column_names)

# DEFINIAMO IL DATALOADER

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
#for batch in train_dataloader: # CHECK CORRECTENESS OF DATA
#    break
#{k: v.shape for k, v in batch.items()}


# instanzion accelerate
accelerator = Accelerator()

# INSTANZIO IL MODELLO

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#     outputs = model(**batch)   # CONTROLLLO CHE TUTTO È OK passando il batch (batch da 8, 2 logits)
#print(outputs.loss, outputs.logits.shape)


            #### CREIAMO IL TRAINING ####

# OTTIMIZZATORE E LR SCHEDULER, COME QUELLI DEL TRAINING USATO SOPRA
optimizer = AdamW(model.parameters(), lr=5e-5)


# SCELGO LA gpu COME DEVICE  (rimuovo)
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# print(device)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
     train_dataloader, eval_dataloader, model, optimizer
 )


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
#print(num_training_steps)


# THE LOOP TRAINING
progress_bar = tqdm(range(num_training_steps))
start = time.time()
model.train()   # SETTA IL MODELLO PER ESSERE TRAINATO
for epoch in range(num_epochs):   # TUTTO IL DATALOADER PER TOT EPOCHE
    for batch in train_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}   # rimuovo
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward() # rimuovo
        accelerator.backward(loss) # (aggiungo)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

end = time.time()
print(end - start)


# THE EVALUAZTION LOOP 
metric = evaluate.load("glue", "mrpc") # QUELLE STANDARD PER QUEL TASK
model.eval() # SETTA IL MODELLO PER FARE INFERENZA (NO DROPOUT / BATCH NORMALIZATION)

eval_dataloader = accelerator.prepare(eval_dataloader)
for batch in eval_dataloader:
    # batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # metric.add_batch(predictions=predictions, references=batch["labels"])
    metric.add_batch(
        predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"])
    )

print(metric.compute())