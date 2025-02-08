import json
import os
import logging
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import torch
from torch.utils.data import Dataset,DataLoader,random_split
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline,get_scheduler
from sacrebleu.metrics import BLEU

# from modelscope.models import Model 
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope.models.nlp import T5ForConditionalGeneration
# from modelscope.preprocessors import TextGenerationTransformersPreprocessor
# 换用huggingface的原库了

def download_model():
    generator = pipeline("text2text-generation", model="Langboat/mengzi-t5-base")
    results = generator(
        "中国的首都位于<extra_id_0>。",
)
    print(results)
    #{'text': '北京'}
    
    tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
    return model,tokenizer


class T5_dataset(Dataset):
    def __init__(self,path):
        super().__init__()
        self.path = path
        self.data = []    
        with open(path,"r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
def collote_fn(batch_samples,tokenizer):
    batch_X,batch_decoder_input ,batch_labels,reference = [],[],[],[]
    for sample in batch_samples:

        batch_X.append("问题："+sample['question'] + "原文："+sample['context'])
        batch_decoder_input.append(f"{sample['answer']}</s>")
        batch_labels.append(f"{sample['answer']}</s>")
        reference.append(f"{sample['answer']}")
        #参考huggingface的构造输入
        # batch_y.append(sample['answer'])
    X = tokenizer(
        batch_X,
        max_length = 512,
        padding = True,
        truncation=True, 
        return_tensors="pt"
    )
    decoder_inputs = tokenizer(
        batch_decoder_input,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )    
    labels = tokenizer(
        batch_labels,
        max_length = 512,
        padding = True,
        truncation=True, 
        return_tensors="pt"
    ) 
    decoder_input_ids = []
    for i in range(len(decoder_inputs['input_ids'])):
        decoder_input_ids.append(torch.cat([
            torch.tensor([tokenizer.pad_token_id]),
            decoder_inputs["input_ids"][i][:-1],
            torch.tensor([0] * (512 - len(decoder_inputs['input_ids'][i])))
        ]))


    labels_with_ignore_index = []
    for i in range(len(labels['input_ids'])):
      labels_with_ignore_index.append(torch.cat([
          labels['input_ids'][i],
          torch.tensor([0] * (512 - len(labels['input_ids'][i])))
      ]))


    return {
        "input_ids": X["input_ids"],
        "attention_mask": X["attention_mask"],
        "decoder_input_ids": torch.stack(decoder_input_ids),
        "labels": torch.stack(labels_with_ignore_index),
        "reference": reference
    }

def eval_bleu(bleu, candidates, references):
    score = bleu.corpus_score(candidates, references)
    return score.score


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO，表示 INFO 及以上级别的日志都会被记录
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志输出格式
        filename="train.log",  # 将日志保存到名为 train.log 的文件中
        filemode="w"  # 使用写入模式打开文件，每次运行都会覆盖之前的内容
    )
    logger = logging.getLogger(__name__)
    batch_size = 4
    epoch = 20
    warmup_ratio = 0.06
    train_data_radio = 0.8
    model,tokenizer = download_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    train_data = T5_dataset("train.json")
    dev_dataset = T5_dataset("dev.json")

    train_size = int(train_data_radio * len(train_data))
    val_size = len(train_data) - train_size

    train_dataset ,val_dataset = random_split(train_data,[train_size,val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=lambda batch: collote_fn(batch, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=lambda batch: collote_fn(batch, tokenizer))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,collate_fn=lambda batch: collote_fn(batch, tokenizer))
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5, weight_decay=0.01)
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps =epoch * num_update_steps_per_epoch
    warm_steps = int(warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    bleu = BLEU()
    model.to(device)
    loss_list = []
    train_bleu,val_bleu,dev_bleu = [],[],[]
    best_bleu = 0.0  # 初始化最佳 BLEU 分数
    best_model_path = "best_model.pth" 
    model.train()
    for epoch in range(1,epoch+1):
        print(f"##### epoch {epoch} #####")
        logger.info(f"##### epoch {epoch} #####")
        t_b,v_b,d_b,l_l = [],[],[],[]
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} (Train)"):
            input_ids = batch['input_ids'].to(device)  # 将数据移动到 GPU
            attention_mask = batch['attention_mask'].to(device)  # 将数据移动到 GPU
            decoder_input = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids= decoder_input,
                        labels=labels
                    )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            l_l.append(float(loss.cpu().detach()))
            out = tokenizer.batch_decode(torch.argmax(outputs.logits, dim=-1), skip_special_tokens=True)
            reference = [[ref] for ref in batch["reference"]]
            batch_bleu_scores = eval_bleu(bleu,out,reference)
            t_b.append(batch_bleu_scores)
        train_bleu.append(round(np.array(t_b).mean(),6))
        loss_list.append(round(np.array(l_l).mean(),6))
        logger.info("train_bleu")
        logger.info(train_bleu)
        logger.info("loss_list")
        logger.info(loss_list)
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            model.eval()
            with torch.no_grad(): 
                for batch in val_loader:
                    outputs = model.generate(input_ids=batch['input_ids'].to(device),attention_mask = batch['attention_mask'].to(device))
                    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    reference = [[ref] for ref in batch["reference"]]
                    batch_bleu_scores = eval_bleu(bleu,out,reference)
                    v_b.append(batch_bleu_scores)
                val_bleu.append(round(np.array(v_b).mean(),6))
                logger.info("val_bleu")
                logger.info(val_bleu)
                if np.array(v_b).mean() > best_bleu:
                    best_bleu = np.array(v_b).mean()
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Epoch {epoch} - New best model saved to {best_model_path} with BLEU score: {best_bleu}")
                    logger.info(f"Epoch {epoch} - New best model saved to {best_model_path} with BLEU score: {best_bleu}")

                for batch in dev_loader:
                    outputs = model.generate(input_ids=batch['input_ids'].to(device),attention_mask = batch['attention_mask'].to(device))
                    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    reference = [[ref] for ref in batch["reference"]]
                    batch_bleu_scores = eval_bleu(bleu,out,reference)
                    d_b.append(batch_bleu_scores)
                dev_bleu.append(round(np.array(d_b).mean(),6))
                logger.info('dev_bleu')
                logger.info(dev_bleu)
            model.train()




