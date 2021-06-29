# -*- coding: utf-8 -*-
import urllib.request
from builtins import *

import requests as re
import urllib3 as url
from bs4 import BeautifulSoup

#get pages
baseurl = "https://movie.douban.com/subject/"
datalist = getdata(baseurl)
#get data
def askURL(url):
	request = urllib.request.Request(url)
	try:
		response = urllib.request.urlopen(request)
		html = response.read()
	except url.error.URLError as e:
		pass
	return html

def getdata(baseurl):
	for i in range(10):
		url = baseurl+str(i)
		html = askURL(url)

	datalist = []
	#anylize data
	return datalist
#post request

#get pages

#anylize pages
'''
three things:
items:
users:user id,user whtched movies, time, rate
users followees:
'''


#save information
def saveData(savepath):
	pass
