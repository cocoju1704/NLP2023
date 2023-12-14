import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
import sys
sentiments_list = ['anger', 'sadness', 'neutral', 'joy', 'admiration']
choose_dataset = "same_num_of_sent_data"




if __name__ == '__main__':
    csvfile = sys.argv[1]
    raw_df = pd.read_csv("./" + choose_dataset + "/test_set.csv")
    num_of_tuple = len(raw_df)

    model_path = "./best_model_" + choose_dataset
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    model.eval()

    res = np.zeros((5, 5))

    for i in range(len(raw_df)):
        encoded = tokenizer(raw_df['text'][i], padding="max_length", truncation=True, max_length=64,
                            return_tensors='pt')
        output = model(**encoded)
        logits = output.logits
        pred = torch.argmax(softmax(logits, dim=1)).item()
        res[raw_df['label'][i]][pred] += 1

    print("pred", end=" ")
    for i in range(5):
        print(sentiments_list[i], end=" ")
    print()
    print("actu")
    for i in range(5):
        print(sentiments_list[i], " ", res[i, :])
    sum_of_correct = 0
    for i in range(5):
        sum_of_correct += res[i, i]
    print("accuracy: ", sum_of_correct / num_of_tuple)