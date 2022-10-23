import requests
import hashlib
# 百度翻译
appid = ''
transalete_key = ''
# 服务器地址
adress = ''
# pastbin密钥
pastbin_key = ''


def transalete(text, appid, transalete_key):
    translate_data = {
        'q': text,
        'from': 'jp',
        'to': 'zh',
        'appid': appid,
        'salt': '260817',
        'sign': hashlib.md5()
    }
    requests.post(
        'https://fanyi-api.baidu.com/api/trans/vip/translate', data=translate_data)


def post_qq_guild(time, result_text):
    result = time + result_text
    response = requests.get(
        f'http://{adress}:5700/send_guild_channel_msg?guild_id=73537411657939073&channel_id=8852195&message={result}')
    print(response.text)
