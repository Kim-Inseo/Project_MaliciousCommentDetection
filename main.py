from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from preprocessing import preprocess_text
from preparing_nlp import prepare_nlp
from modeling import classify

class Item(BaseModel):
    text: List[str] = Field(examples=['도전을 안 하는 것만큼 무의미한 것은 없어'])

app = FastAPI()

@app.get('/')
async def welcome():
    return {'message': 'Welcome to Malicious Comments Detection API'}

@app.post('/predict')
async def classify_comment(item: Item):
    after_preprocess = preprocess_text(item.text)
    after_prepare = prepare_nlp(after_preprocess)
    after_classify = classify(after_prepare)

    result_list = {}

    for i in range(len(item.text)):
        result = {}
        result['text'] = item.text[i]
        result['악성 댓글 여부'] = after_classify[i][0]
        result['확률'] = '{:.2f}%'.format(after_classify[i][1].item() * 100)

        result_list[f'{i+1}번째 문장'] = result

    return result_list