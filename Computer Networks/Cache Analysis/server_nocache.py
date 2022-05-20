import socket
import sys   
import re 
import time   
import requests

cache = dict()
response_message =''


def get_article_from_server(url):
    global response_message
    res = None
    try:
        response = requests.get(url)
        response_message = "MISS"
        res = response.text
    except:
        response_message = '404 NOT FOUND'
    return res

def get_article(url):
    global response_message

    if (url.startswith("www.")):
        url = "http://"+url
    elif not (url.startswith("http://www.")):
        url = "http://www."+url
    print(url)
    return get_article_from_server(url)



def server_program(port):
    global response_message
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        
    print ("Socket successfully created")

    host = socket.gethostname()
    print("host: "+host)
    server_address = ('', port)

    s.bind(server_address)
         
    print("socket binded to %s" %port)
    
    # put the socket into listening mode
    s.listen(5)    
    print("socket is listening")

    # the program keep  running waiting for connection from client 
    while True:
        print('waiting for connection ...')
        
      

        # server recieved connection from client 
        c, addr = s.accept()    
        print ('Got connection from', addr )
        # a forever loop until we interrupt it or
        # an error occurs
        while True:
            try:
                completeData = ''
                # loop until all data is recived. siince the server only recieve 256 byte
                # some data is much bigger than that
                while True:
                    data = c.recv(1024)
                    if data:
                        completeData = completeData + data.decode()  
                        if(len(data) < 1024):
                            break     
                       
                    else:
                        print ("No data")
                        resp = '404 ERROR' # if no data recieved, send error to client and close connection                  
                        break           
                res = get_article(completeData) 
                if res:
                    #print("HIT or MISS ",response_message)
                    response_message = response_message + res
                    #print("HIT or MISS ",response_message)

                c.sendall(response_message.encode('utf-8'))
                c.close()
                break

            except:
                print('close connection because of error')
                c.sendall(resp.encode('utf-8')) # send error message and close connection 
                break
               
            
            finally:
                print('close connection')


if __name__ == '__main__':
    
    port = 9000
    server_program(port)
    
    