from konlpy.tag import Okt
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

corr_model = T5ForConditionalGeneration.from_pretrained('j5ng/et5-typos-corrector')
tokenizer = T5Tokenizer.from_pretrained('j5ng/et5-typos-corrector')

okt = Okt()
stopwords = ['하다']

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ",
                 "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"',
                 '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha',
                 '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }


def clean(text, punct, mapping):
    '''
    :param text: 한국어 문장
    :param punct: 문장 부호, 기호 등등을 포함한 문자열
    :param mapping: 각각의 기호를 대체하기 위한 사전
    :return: 기호가 대체된 한국어 문장
    '''
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


def clean_str(text):
    '''
    :param text: 기호가 대체된 한국어 문장
    :return: 텍스트 클리닝이 완료된 한국어 문장
    '''
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수 기호 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
    text = re.sub('\n', '.', string=text)
    return text

# 맞춤법 수정 함수
def spell_correction(text_list, corr_model, tokenizer):
    '''
    :param text_list: 1차원 list(각 원소는 한국어 문장)
    :param corr_model: 사전 훈련된 모델
    :param tokenizer: 사전 훈련된 모델
    :return: 1차원 list(각 원소는 맞춤법이 수정된 한국어 문장)
    '''
    output_text_list = []

    for text in text_list:
        input_encoding = tokenizer("맞춤법을 고쳐주세요: " + text, return_tensors="pt")

        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask

        output_encoding = corr_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )

        output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)
        output_text_list.append(output_text)

    return output_text_list

# 토큰화 함수 작성
def tokenize(text_list, analyzer, stopwords):
    '''
    :param text_list: 1차원 list(각 원소는 맞춤법이 수정된 한국어 문장)
    :param analyzer: 한국어 형태소 분석기(Okt)
    :param stopwords: 한국어 불용어 list
    :return: 2차원 list
            (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
    '''
    output_tokens_list = []

    for text in text_list:
        pos_token = analyzer.pos(text, stem=True, norm=True)

        after_pos = []
        exclude_tag_list = ['Josa', 'PreEomi', 'Eomi', 'Punctuation', 'Foreign']

        for token, tag in pos_token:
            if tag not in exclude_tag_list:
                after_pos.append(token)

        tokens = [word for word in after_pos if word not in stopwords]
        output_tokens_list.append(tokens)

    return output_tokens_list

# 전처리 작업 수행
def preprocess_text(text_list):
    '''
    :param text_list: 1차원 list(각 원소는 한국어 문장)
    :return: 2차원 list
            (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)

    spell_correction(), tokenize()를 한 함수로 묶음
    '''
    cleaned_text_list = []
    for text in text_list:
        cleaned_text = clean(text, punct, punct_mapping)
        cleaned_text = clean_str(cleaned_text)
        cleaned_text_list.append(cleaned_text)
    corr_text_list = spell_correction(cleaned_text_list, corr_model, tokenizer)
    output_tokens_list = tokenize(corr_text_list, okt, stopwords)

    return output_tokens_list



