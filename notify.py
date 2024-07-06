import requests

def pushplus(title,content):
    data = {
        "token" : '5fa3024cf8564bc5aacab342244faf21',
        "title" : title,
        "content" : content,
        "template": "markdown"
    }
    requests.post("http://www.pushplus.plus/send",data)   
    
if __name__ =="__main__":
    pushplus('a','b')