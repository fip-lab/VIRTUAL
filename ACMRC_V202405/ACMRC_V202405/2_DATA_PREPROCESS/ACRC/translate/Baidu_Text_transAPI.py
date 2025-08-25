# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = '20221104001435188'
appkey = 'dPPRPlnE3cMzN3dxN1vH'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'zh'
to_lang =  'wyw'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def baidu_trans(query):
    from_lang = 'wyw'
    to_lang = 'zh'
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    return_list = []
    try:
        for r in result['trans_result']:
            return_list.append(r['dst'])
    except:
        print(result)

    #print(return_list[0])
    return return_list[0]
    # Show response
    #print(json.dumps(result, indent=4, ensure_ascii=False))

q = '燕荣，字贵公，华阴弘农人也。父偘，周大将军。'
baidu_trans(q)
print("#####")