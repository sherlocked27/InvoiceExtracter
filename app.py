from fastapi import FastAPI, File, UploadFile
import uvicorn
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import torch
from torch.utils.data import Dataset, DataLoader
from icecream import ic
from app.dataset_creator import create_pred_df, InvoicePredictSet, predict_fn, save_labelled
from unilm.layoutlmft.layoutlmft.models.layoutlmv2.tokenization_layoutlmv2 import LayoutLMv2Tokenizer
from unilm.layoutlmft.layoutlmft.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from unilm.layoutlmft.layoutlmft.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2ForTokenClassification


def get_labels_all_old():
    with open('/home/ubuntu/invoice_inference_api/labels_all_old.txt') as f:
        labels = f.read().split('\n')
    return labels


def get_labels_all():
    with open('/home/ubuntu/invoice_inference_api/labels_other10_2.txt') as f:
        labels = f.read().split('\n')
    return labels


id2label_all_old = {v: k for v, k in enumerate(get_labels_all_old())}
label2id_all_old = {k: v for v, k in enumerate(get_labels_all_old())}

id2label_all = {v: k for v, k in enumerate(get_labels_all())}
label2id_all = {k: v for v, k in enumerate(get_labels_all())}

model_path = 'microsoft/layoutlmv2-base-uncased'
tokenizer = LayoutLMv2Tokenizer.from_pretrained(model_path)
config_all_old = LayoutLMv2Config.from_pretrained(model_path, num_labels=len(
    get_labels_all_old()), id2label=id2label_all_old, label2id=label2id_all_old)
config_all = LayoutLMv2Config.from_pretrained(model_path, num_labels=len(
    get_labels_all()), id2label=id2label_all, label2id=label2id_all)
trained_model_all_old = LayoutLMv2ForTokenClassification.from_pretrained(
    '/home/ubuntu/New_annotated_data/model_raw/model.bin', config=config_all_old)
trained_model_all = LayoutLMv2ForTokenClassification.from_pretrained(
    '/home/ubuntu/New_annotated_data/model_raw_other10_part2/model.bin', config=config_all)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
trained_model_all_old.to(device)
trained_model_all.to(device)

app = FastAPI()


@app.post("/extractInfo/")
async def extract_info(image: UploadFile = File(...)):
    image = Image.open(image.file)
    image.save("input.jpg")
    # ic()
    pred_df = create_pred_df(image)
    # ic()
    if(len(pred_df) == 0):
        return {"Message": "OCR fucked up, Sorry"}
    predict_dataset = InvoicePredictSet(
        df=pred_df, tokenizer=tokenizer, max_length=512, target_size=224, image=image)
    # ic()
    predict_dataloader = DataLoader(predict_dataset, batch_size=9)
    pred_out_all_old = predict_fn(eval_dataloader=predict_dataloader,
                                  model=trained_model_all_old, id2label=id2label_all_old, tokenizer=tokenizer)
    pred_out_all = predict_fn(eval_dataloader=predict_dataloader,
                              model=trained_model_all, id2label=id2label_all, tokenizer=tokenizer)
    ic(pred_out_all_old)
    ic(pred_out_all)
    for key, value in pred_out_all['words'].items():
        pred_out_all_old['words'][key] = value
        pred_out_all_old['bbox'][key] = pred_out_all['bbox'][key]

    width, height = image.size
    pred_out_all_old['width'] = width
    pred_out_all_old['height'] = height

    save_labelled(pred_out_all_old)
    return pred_out_all_old

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
