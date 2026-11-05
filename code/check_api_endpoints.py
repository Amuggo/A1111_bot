import bootstrap_env  # noqa: F401,E402  — ДОЛЖЕН идти первым: переключает скрипт на библиотеки из папки env/
import urllib.request
import json
import os

webui_server_url = os.environ.get('SD_BASE_URL', 'http://127.0.0.1:7860').rstrip('/')

# Попробуем разные варианты эндпоинтов
endpoints_to_try = [
    'sdapi/v1/txt2img',
    'sdapi/v1/img2img', 
    'sdapi/v1/sd-models',
    'sdapi/v1/samplers',
    'api/txt2img',
    'txt2img',
    'run/predict',
    'api/predict',
]

for endpoint in endpoints_to_try:
    try:
        url = f'{webui_server_url}/{endpoint}'
        request = urllib.request.Request(url, method='GET')
        response = urllib.request.urlopen(request, timeout=5)
        print(f'✅ {endpoint} - доступен (статус: {response.status})')
    except urllib.error.HTTPError as e:
        print(f'❌ {endpoint} - {e.code}: {e.reason}')
    except Exception as e:
        print(f'❌ {endpoint} - ошибка: {e}')