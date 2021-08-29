import numpy as np
import pytesseract
import torch
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from icecream import ic
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
def get_labels():
    with open('/home/ubuntu/invoice_inference_api/labels.txt') as f:
        labels = f.read().split('\n')
    return labels

id2label = {v: k for v, k in enumerate(get_labels())}
label2id = {k: v for v, k in enumerate(get_labels())}

def normalize_box(box, width, height):
    width = int(width)
    height = int(height)
    return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def resize_and_align_bounding_box(bbox, original_image, target_size):
    x_, y_ = original_image.size
    x_scale = target_size / x_ 
    y_scale = target_size / y_
    origLeft, origTop, origRight, origBottom = tuple(bbox)
    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale)) 
    return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]

def create_pred_df(image):
    tess_df = pytesseract.image_to_data(image,output_type='data.frame')
    tess_df = tess_df[tess_df.conf > 0]
    tess_df = tess_df[tess_df.text.str.strip() != '']
    tess_df['x0'] = tess_df.apply(lambda row: row['left'], axis=1)
    tess_df['y0'] = tess_df.apply(lambda row: row['top'], axis=1)
    tess_df['x2'] = tess_df.apply(lambda row: row['left']+row['width'], axis=1)
    tess_df['y2'] = tess_df.apply(lambda row: row['top']+row['height'], axis=1)
    tess_df['bbox'] = tess_df[['x0','y0','x2','y2']].values.tolist()
    tess_df = tess_df[['bbox','text']]
    tess_df['imageWidth'], tess_df['imageHeight'] = image.size
    pred_tess_df = tess_df.groupby(['imageWidth','imageHeight']).agg(list).reset_index()
    return pred_tess_df

class InvoicePredictSet(Dataset):
    """LayoutLM dataset with visual features."""

    def __init__(self, df, tokenizer, max_length, target_size, image):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.target_size = target_size
        self.pad_token_box = [0, 0, 0, 0]
        self.train = False
        self.image = image

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx,:].to_dict()        
        #base_path = data_config.base_image_path
        original_image = self.image.convert("RGB")
        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))
        # first, read in annotations at word-level (words, bounding boxes, labels)
        words = item["text"]
        unnormalized_word_boxes = item["bbox"]
#         word_labels = item["label"]
        width = item["imageWidth"]
        height = item["imageHeight"]
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        assert len(words) == len(normalized_word_boxes)

        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        token_boxes = []
        unnormalized_token_boxes = []
#         token_labels = []
        for word, unnormalized_box, box in zip(words, unnormalized_word_boxes, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # ******** Try this approch too---
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            
#             for i in range(len(word_tokens)):
               
#                 token_labels.extend([label])
        
        # Truncation of token_boxes + token_labels
        special_tokens_count = 2 
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
#             token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]
        
        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [self.pad_token_box] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [self.pad_token_box] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
#         token_labels = [-100] + token_labels + [-100]
        
        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
#         token_labels += [-100] * padding_length
        encoding['bbox'] = token_boxes
        encoding['unnormalized_bbox'] = unnormalized_token_boxes
#         encoding['labels'] = token_labels

        assert len(encoding['input_ids']) == self.max_seq_length
        assert len(encoding['attention_mask']) == self.max_seq_length
        assert len(encoding['token_type_ids']) == self.max_seq_length
        assert len(encoding['bbox']) == self.max_seq_length
#         assert len(encoding['labels']) == self.max_seq_length

        encoding['resized_image'] = ToTensor()(resized_image)
        # rescale and align the bounding boxes to match the resized image size (typically 224x224) 
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size) for bbox in unnormalized_token_boxes]
        #encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
        
        # finally, convert everything to PyTorch tensors 
        for k,v in encoding.items():
#             if k == 'labels':
#                 label_indices = []
#                 # convert labels from string to indices
#                 for label in encoding[k]:
#                     if label != -100:
#                         label_indices.append(label2id[label])
#                     else:
#                         label_indices.append(label)
#                 encoding[k] = label_indices
            encoding[k] = torch.as_tensor(encoding[k])
        return encoding

def get_merged_bbox(bboxes):
    min_x0 = math.inf
    min_y0 = math.inf
    max_x2 = -math.inf
    max_y2 = -math.inf
    bboxes = bboxes.tolist()
    for bbox in bboxes:
        bbox = bbox.tolist()
        min_x0 = min(min_x0,bbox[0])
        min_y0 = min(min_y0,bbox[1])
        max_x2 = max(max_x2,bbox[2])
        max_y2 = max(max_y2,bbox[3])
    return [min_x0,min_y0,max_x2,max_y2]
        

def predict_fn(eval_dataloader, model, tokenizer):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    input_id = []
    bboxes = []
    tk0 = tqdm(eval_dataloader, total = len(eval_dataloader))
    for bi, batch in enumerate(tk0):
        with torch.no_grad():
            input_ids=batch['input_ids'].to(device)
            bbox=batch['bbox'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device)
#             labels=batch['labels'].to(device)
            resized_images = batch['resized_image'].to(device) 
            resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device)
            outputs = model(image = resized_images,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
#             tmp_eval_loss = outputs.loss
            logits = outputs.logits
#             eval_loss += tmp_eval_loss.item()
#             nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
#                 out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#                 out_label_ids = np.append(
#                     out_label_ids, labels.detach().cpu().numpy(), axis=0
#                 )
            input_id.append(batch['input_ids'].numpy())
            bboxes.append(batch['unnormalized_bbox'].numpy())
#     eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)
    preds = [[id2label[p] for p in pred] for pred in preds]
    words = tokenizer.convert_ids_to_tokens(input_id[0][0])

    out = pd.DataFrame(list(zip(words,preds[0],bboxes[0][0])),columns=['words','label','bbox'])
    # out = pd.DataFrame(list(zip(words,preds[0])),columns=['words','label'])
    # ic(out)
    # print(out)
    out = out[out['label'] != 'O']
    # ic(out)
    label_grps = out.groupby(by='label').agg({'words':' '.join,'bbox': get_merged_bbox})
    # ic(label_grps)
    # label_grps = out.groupby(by='label').agg({'words':' '.join})

    # ic(label_grps)
    try:
        label_grps['words'] = label_grps['words'].str.replace(' ##','')
        return label_grps.to_dict()
    except:
        return {'bbox':{},'words':{}}
    # print(label_grps)


    # for i,pred in enumerate(preds[0]):
    #     ic(words[i],id2label[pred])

    # return zip(words,bboxes[0][0],preds[0])


