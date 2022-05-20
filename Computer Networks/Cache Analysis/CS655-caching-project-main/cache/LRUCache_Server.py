import requests
import time
import socket

#Define each node of the cache
class Node :

    def __init__(self, key, value) :

        self.key = key
        self.value = value
        self.next = None
        self.previous = None

#Implementing the cache as a doubly linked list 
class LRUCache :

    cacheLimit = 200

    def __init__(self, function) :

        self.function = function
        self.cache = {}
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.previous  = self.head
        

    def __call__(self, *args, **kwargs) :

        if args in self.cache :
            self.linkedList(args)
            return "HIT ".encode()+self.cache[args]

        if len(self.cache) > self.cacheLimit :

            n = self.head.next
            self._remove(n)
            del self.cache[n.key]

        content = self.function(*args, **kwargs)
        self.cache[args] = content
        node = Node(args, content)
        self._add(node)

        print("cache size ", len(self.cache))

        return content

    def _remove(self, node) :

        p = node.previous
        n = node.next
        p.next = n
        n.previous = p
    
    def _add(self, node) : 

        p = self.tail.previous
        p.next = node
        self.tail.previous = node
        node.previous = p
        node.next = self.tail

    def linkedList(self, args)  :

        current = self.head

        while True :

            if current.key == args :

                node = current
                self._remove(node)
                self._add(node)
                del self.cache[node.key]
                self.cache[node.key] = node.value
                break

            else :

                current = current.next

#Function to fetch the http page
@LRUCache
def fetchArticle(url):
    response = requests.get(url)
    content = response.content

    return "MISS ".encode()+content


def server_program(port):
    
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
                url = ''
                # loop until all data is received. siince the server only recieve 256 byte
                # some data is much bigger than that
                while True:
                    data = c.recv(1024)
                    if data:
                        url = url + data.decode()  
                        if(len(data) < 1024):
                            break     
                       
                    else:
                        print ("No data")
                        resp = '404 ERROR' # if no data recieved, send error to client and close connection                 
                        break
                if (url.startswith("www.")):
                    url = "http://"+url
                elif not (url.startswith("http://www.")):
                    url = "http://www."+url

                res = fetchArticle(url)

                if res:
                    response_message = res

                c.sendall(response_message)
                c.close()
                break

            except:
                print('close connection because of error')
                resp = '404 ERROR'
                c.sendall(resp.encode()) # send error message and close connection 
                c.close()
                break
               
            
            finally:
                print('close connection')



if __name__ == '__main__':
    
    port = 9000
    server_program(port)