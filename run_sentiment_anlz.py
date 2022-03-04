import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src import config
from src.processor import load_data, preprocess, rm_commom_words, vocab_mapping, balance_classes
from src.dataset import SentimentDataset
from src.model import SentimentClassifier
from src.optimizer import AdamW
from src.lr_scheduler import get_linear_schedule_with_warmup


import importlib
importlib.reload(config)
import src
importlib.reload(src)


def train_fcn(train_dataloader, valid_dataloader, train_batch_size, valid_batch_size,
             model, optimizer, scheduler, device, grad_clip, epoch, steps, every_n_step):
    model.train()

    valid_steps = len(valid_dataloader)

    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch}",
                              leave=False, disable=False)

    # get a batch of data dict
    for data in train_progress_bar:

        steps += 1
        hidden = model.init_hidden(batch_size=train_batch_size)

        for k, v in data.items():
            data[k] = v.to(device)

        for h in hidden:
            h.to(device)

        train_logits, hidden, train_loss = model(hidden, **data)

        optimizer.zero_grad()
        train_loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if steps % every_n_step == 0:
            model.eval()

            total_valid_loss = 0

            for data in valid_dataloader:

                val_hidden = model.init_hidden(valid_batch_size)

                for k, v in data.items():
                    data[k] = v.to(device)

                for h in val_hidden:
                    h.to(device)

                # val_hidden = tuple([each.data for each in val_hidden])

                with torch.no_grad():
                    valid_logits, val_hidden, valid_loss = model(val_hidden, **data)

                total_valid_loss += valid_loss.item()

                # (batch_size, output_size)
                output_ps = torch.exp(valid_logits)
                # (batch_size)
                pred = np.argmax(output_ps.detach().cpu().numpy(), axis=1)

                labels = data['labels'].detach().cpu().numpy()
                num_correct = sum(pred == labels)
                accuracy = num_correct / len(labels)







def main():
    # load the json file and extract messages and sentiments
    messages, sentiments = load_data(config.TRAINING_FILE)
    num_messages = len(messages)
    uniq_sentiment = len(set(sentiments))

    # messages[0], sentiments[0]

    # Process all the twits to tokenized_msg: List[List[str]]
    tokenized_msgs = [preprocess(m) for m in messages]

    # Create a vocabulary by using Bag of words
    bag_of_words = Counter([word for sent_lst in tokenized_msgs for word in sent_lst])

    # remove most common words such as 'the', 'and', etc. and rare words
    filtered_words = rm_commom_words(bag_of_words, num_messages, config.LOW_CUTOFF, config.HIGH_CUTOFF)

    # Build the vocab mapping with filtered_words
    vocab_to_id, id_to_vocab = vocab_mapping(filtered_words,
                                             cls_id=config.CLS_ID,
                                             sep_id=config.SEP_ID,
                                             pad_id=config.PAD_ID)

    # tokenized with the words not in `filtered_words` removed.
    filtered_msgs = [[word for word in token_msg if word in vocab_to_id] for token_msg in tokenized_msgs]
    # print(tokenized_msgs[0], '\n', filtered_msgs[0])

    # Balance the neutral class
    balanced_dict = balance_classes(filtered_msgs, sentiments, config.NEUTRAL_SCORE)

    # train-validation-split
    train_features, val_features, train_labels, val_labels = train_test_split(
        balanced_dict['messages'],
        balanced_dict['sentiments'],
        test_size=config.DATA_SPLIT_RATIO,
        random_state=42,
        stratify=balanced_dict['sentiments'])

    train_dataset = SentimentDataset(
        texts=train_features,
        labels=train_labels,
        vocab_to_id=vocab_to_id,
        cls_id=config.CLS_ID,
        sep_id=config.SEP_ID,
        max_len=config.MAX_LEN,
        pad_on_right=True
    )

    valid_dataset = SentimentDataset(
        texts=val_features,
        labels=val_labels,
        vocab_to_id=vocab_to_id,
        cls_id=config.CLS_ID,
        sep_id=config.SEP_ID,
        max_len=config.MAX_LEN,
        pad_on_right=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentimentClassifier(len(vocab_to_id),
                                config.EMBED_SIZE,
                                config.LSTM_SIZE,
                                config.OUTPUT_SIZE,
                                config.LSTM_LAYER,
                                config.DROP_RATE,
                                uniq_sentiment)
    model.embedding.weight.data.uniform_(-1, 1)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)

    NUM_TRAIN_STEPS = int(len(train_dataloader) * config.EPOCHS)
    NUM_WARMUP_STEPS = int(NUM_TRAIN_STEPS * config.WARMUP_PROPORTION)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=NUM_WARMUP_STEPS,
                                                num_training_steps=NUM_TRAIN_STEPS,
                                                last_epoch=-1)




