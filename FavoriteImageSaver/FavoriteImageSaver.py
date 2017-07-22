# coding: utf-8
import tweepy
import traceback
import time
from urllib.request import urlopen

#保存したいフォルダのパスを入れてくれ！！！ここを入力しないとこのファイルがあるところに大量の画像が保存されてしまうぞ！！！！
FolderPath=""

#誰のファボ欄の画像を保存したいのか""の間に書いてくれ！！「@~~」ってやつだぞ！！！複数人いるなら["@~~","@=="]みたいな感じで書いてくれ！！！
bonnous = [""]

#twitter認証だ！Twitter Developperとかいうところに登録してコンシューマーキー、コンシューマーシークレット、アクセストークン、アクセスシークレットを手に入れてここに入力だ！！！！！！
CK = ""
CS = ""
AT = ""
AS = ""

def on_status(status):
    try:
        if status.extended_entities['media']!=[]:
            i=0
            for media in status.extended_entities['media']:
                url = media['media_url_https']
                url_orig = '%s:orig' %  url
                filename = url.split('/')[-1]
                savepath = FolderPath + str(i) +filename
                i+=1
                response = urlopen(url_orig)
                with open(savepath, "wb") as f:
                    f.write(response.read())         
        
    except Exception as e:
        print("[-] Error: ", e)
    return True
     



auth = tweepy.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)
api = tweepy.API(auth)

for name in bonnous:
    for i in range(1,220):
    	try:
    		print(i)
    		favs = api.favorites(id=name,page=(i))
    		print(len(favs))


    		for status in favs:
    			on_status(status)
    	except Exception as e:
    		print("[-] Error: ", e)
    		time.sleep(15*60+1)
    		i -= 1
