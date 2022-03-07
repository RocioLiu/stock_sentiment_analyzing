import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src import config
from src.processor import load_data, preprocess, rm_commom_words, vocab_mapping, balance_classes
from src.dataset import SentimentDataset
from src.model import SentimentClassifier
from src.optimizer import AdamW
from src.lr_scheduler import get_linear_schedule_with_warmup


# import importlib
# importlib.reload(config)
# import src
# importlib.reload(src)


def training_fn(train_dataloader, val_dataloader, train_batch_size, val_batch_size,
                model, optimizer, scheduler, device, grad_clip, epoch, steps,
                every_n_step, history_dict):

    model.train()

    # val_steps = len(val_dataloader)

    # train_progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch}",
    #                           leave=False, disable=False)

    # get a batch of data dict
    for data in tqdm(train_dataloader, position=0, leave=True):

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

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, '\n', param.size(), '\n', param.data, '\n\n')

        if steps % every_n_step == 0:
            model.eval()

            y_true_list = []
            y_pred_list = []

            val_losses = []
            val_accuracy = []

            for data in val_dataloader:

                val_hidden = model.init_hidden(val_batch_size)

                for k, v in data.items():
                    data[k] = v.to(device)

                for h in val_hidden:
                    h.to(device)

                # val_hidden = tuple([each.data for each in val_hidden])

                with torch.no_grad():
                    val_logits, val_hidden, val_loss = model(val_hidden, **data)

                val_losses.append(val_loss.item())

                # (batch_size, output_size)
                output_ps = torch.exp(val_logits)
                # (batch_size)
                preds = np.argmax(output_ps.detach().cpu().numpy(), axis=1)

                labels = data['labels'].detach().cpu().numpy()
                num_correct = sum(preds == labels)
                val_accuracy.append(num_correct / len(labels))

                y_true_list.append(labels)
                y_pred_list.append(preds)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_accuracy = sum(val_accuracy) / len(val_accuracy)

            y_true_array = np.concatenate(y_true_list)
            y_pred_array = np.concatenate(y_pred_list)
            val_f1 = f1_score(y_true_array, y_pred_array, average='weighted')

            history_dict['step'].append(steps)
            history_dict['train_loss'].append(train_loss.item())
            history_dict['val_loss'].append(avg_val_loss)
            history_dict['val_acc'].append(avg_val_accuracy)
            history_dict['val_f1'].append(val_f1)

            model.train()

            print(f"\nEpoch: {epoch}/{config.EPOCHS}    step: {steps}")
            print(f"- train_loss: {train_loss.item():.4f} - val_loss: {avg_val_loss:.4f}\n"
                  f"- val_accuracy: {avg_val_accuracy:.4f} - val_f1: {val_f1:.4f} \n")

    # return the last eval_f1 after traverse an epoch
    return steps, val_f1, history_dict


def predict_fn(text, model, vocab_to_id, cls_id, sep_id, device) -> int:
    """
    Make a prediction on a single sentence.
    Args:
        text: str. A message.
    returns:
        pred: the sentiment score of this text
    """
    tokens = preprocess(text)

    # Filter non-vocab words
    filtered_tokens = [w for w in tokens if w in vocab_to_id]

    # Convert words to ids
    token_ids = [cls_id] + [vocab_to_id[w] for w in filtered_tokens] + [sep_id]

    # Adding a batch dimension
    input_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(-1)

    hidden = model.init_hidden(1)

    input_ids.to(device)

    for h in hidden:
        h.to(device)

    # Get the NN output
    with torch.no_grad():
        logps, _, _ = model(hidden, input_ids, None, None)

    output_ps = torch.exp(logps)
    pred = np.argmax(output_ps.detach().cpu().numpy())

    return pred




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
    # return the words that have not been filtered out
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

    val_dataset = SentimentDataset(
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
        shuffle=True, drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=True, drop_last=True
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
                                                num_warmup_steps=0,
                                                num_training_steps=NUM_TRAIN_STEPS,
                                                last_epoch=-1)

    # -- training process --
    best_f1 = 0
    steps = 0
    history_dict = {"step": [],
                    "train_loss": [],
                    "val_loss": [],
                    "val_acc": [],
                    "val_f1": []}

    for epoch in tqdm(range(1, config.EPOCHS + 1)):
        steps, val_f1, history_dict = training_fn(train_dataloader,
                                                val_dataloader,
                                                config.TRAIN_BATCH_SIZE,
                                                config.VAL_BATCH_SIZE,
                                                model, optimizer,
                                                scheduler, device,
                                                config.GRAD_CLIP,
                                                epoch, steps,
                                                config.EVERY_N_STEP,
                                                history_dict)

        if val_f1 > best_f1:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = val_f1

    with open(config.OUTPUT_JSON, 'w') as file:
        json.dump(history_dict, file)


    # -- inference process --
    # Load the trained model for prediction
    model = SentimentClassifier(len(vocab_to_id),
                                config.EMBED_SIZE,
                                config.LSTM_SIZE,
                                config.OUTPUT_SIZE,
                                config.LSTM_LAYER,
                                config.DROP_RATE,
                                uniq_sentiment)
    model.embedding.weight.data.uniform_(-1, 1)
    checkpoint = torch.load(config.MODEL_PATH) # map_location=torch.device('cpu')
    model.load_state_dict(checkpoint)
    model.to(device)

    # with open(config.TEST_FILE, 'r') as json_file:
    #     data_dict = json.load(json_file)
    #
    # test_msgs = [data['message_body'] for data in data_dict['data']]
    # tokenized_test_msgs = [preprocess(msg) for msg in test_msgs]
    # filtered_test_msgs = [[word for word in token_msg if word in vocab_to_id]
    #                       for token_msg in tokenized_test_msgs]


    print(config.MSG1)
    print(f"sentiment: {predict_fn(config.MSG1, model, vocab_to_id, config.CLS_ID, config.SEP_ID, device)}")

    print(config.MSG2)
    print(f"sentiment: {predict_fn(config.MSG2, model, vocab_to_id, config.CLS_ID, config.SEP_ID, device)}")


if __name__ == "__main__":
    main()



