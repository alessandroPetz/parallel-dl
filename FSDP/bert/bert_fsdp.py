import os
import torch
import time
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import wrap
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from datasets import load_dataset, load_from_disk

def setup_fsdp():
    """Setup per FSDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup():
    """Chiude il processo distribuito."""
    dist.destroy_process_group()

def get_dataloaders(batch_size):
    """Prepara i DataLoader distribuiti."""
    dataset_path = "tokenized_datasets"
    
    if not os.path.exists(dataset_path):
        raw_datasets = load_dataset("glue", "mrpc")   #qnli o mrpc
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_function(example):
            return tokenizer(example["question"], example["sentence"], truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        tokenized_datasets.save_to_disk(dataset_path)
    else:
        tokenized_datasets = load_from_disk(dataset_path)

    train_sampler = DistributedSampler(tokenized_datasets["train"])
    eval_sampler = DistributedSampler(tokenized_datasets["validation"], shuffle=False)

    data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
    )

    return train_dataloader, eval_dataloader

def train(rank, num_epochs, batch_size):
    """Esegue il training distribuito con FSDP."""
    setup_fsdp()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
    
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    model.train()
    start_training_time = time.time()

    for epoch in range(num_epochs):
        cont_print = 0
        train_dataloader.sampler.set_epoch(epoch)
        for batch in train_dataloader:
            if cont_print % 10 == 0:
                b_sz = len(batch["attention_mask"])
                print(f"[GPU{device} - RANK{local_rank}] Epoch {epoch} | Batchsize: {b_sz} | cont_print {cont_print}")
            cont_print += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
    end_training_time = time.time()
    print(f"[GPU{device} - RANK{local_rank}] Tempo di training: {end_training_time - start_training_time:.2f} secondi")
    
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with FSDP")
    parser.add_argument("--epochs", default=1, type=int, help="Numero di epoche di training")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU")
    args = parser.parse_args()

    train(rank=int(os.environ["RANK"]), num_epochs=args.epochs, batch_size=args.batch_size)