import requests
import json

url = 'http://127.0.0.1:8000/predict'

data = {
    'text': [
        '도전을 안 하는 것만큼 무의미한 것은 없어',
        '야 이 자식아 너 뭐하냐 죽여버린다',
    ]
}

response = requests.post(url, json=data)

print(json.dumps(response.json(),
                 ensure_ascii=False,
                 indent=4))