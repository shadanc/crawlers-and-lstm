# -*- coding:utf-8 -*-

"""
@ author : XT
@file :test111.py
@time :18-4-19 下午5:38

"""

import urllib.request
import time
import requests
import socks
import socket
import random
import os

#设置代理
socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 1080)
socket.socket = socks.socksocket

#初始化数组
urls=[]
Names=[]
with open("urls.txt", 'r') as f1:
    for line in f1.readlines():
            urls.append(line)
with open("names.txt" ,'r') as f2:
    for line in f2.readlines():
            Names.append(line)



def auto_down(url,filename):
    try:
        urllib.request.urlretrieve(url,filename)
    except urllib.request.ContentTooShortError:
        print ('Network conditions is not good.Reloading.')
        auto_down(url,filename)

for k in range(len(urls)):
    filename = "./img/"+Names[k]
    if os.path.exists(filename):
        continue
    try:
        user_agent="sseas"+str(k)+"es5xf"
        headers = {'User-agent': user_agent}
        url = urls[k]
        print(url)
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent',
                              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        auto_down(url, filename)
        time.sleep(5*random.random()) #做伪装防止屏蔽 幻数可以改！
    except requests.exceptions.ConnectionError as e:
        print('Error', e.args)