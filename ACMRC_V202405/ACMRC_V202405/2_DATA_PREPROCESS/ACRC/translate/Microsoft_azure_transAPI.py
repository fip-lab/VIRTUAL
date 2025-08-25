#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Microsoft_azure_transAPI.py
@Author  : huanggj
@Time    : 2023/2/2 13:49
"""
import requests, uuid, json

# Add your key and endpoint
#key = "72be8b7160b9460d85a11eab4bc6f617"
key = "6b9be80be71b4c3f903e4f77e617f11c"
endpoint = "https://api.cognitive.microsofttranslator.com"

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "eastasia"

path = '/translate'
constructed_url = endpoint + path



def microsoft_trans(query, type):
    from_ = 'lzh' # 文言文
    to_ = 'zh-Hans' # 简体
    if type == 'option':
        from_ = 'zh-Hans'
        to_ = 'lzh'

    params = {
        'api-version': '3.0',
        'from': from_,
        'to': to_
        # 'to': 'zh-Hant' # 繁体
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': query
    }]

    request = None
    while True:
        try:
            request = requests.post(constructed_url, params=params, headers=headers, json=body, timeout=5)
            print('success')
            break
        except requests.exceptions.RequestException as e:
            print(e)
            continue
    #request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    #print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))
    res = response[0]['translations'][0]['text']
    print(res)
    #print(res)
    return res


microsoft_trans("sdd","a")