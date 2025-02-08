import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from train import download_model,collote_fn
from train import T5_dataset

if __name__ =="__main__":
    model,tokenizer = download_model()
    model.load_state_dict(torch.load("best_model.pth"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    dev_dataset = T5_dataset("dev.json")
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False,collate_fn=lambda batch: collote_fn(batch, tokenizer))
    model.to(device)
    model.eval()
    for batch in dev_loader:
        outputs = model.generate(input_ids=batch['input_ids'].to(device),attention_mask = batch['attention_mask'].to(device))
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        reference = batch["reference"]
        print(f"reference:{reference} \n outputs:{outputs}")
