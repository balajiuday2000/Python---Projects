import socket
import sys

#Host address and Port Number are provided as command line arguments 
HOST = sys.argv[1]
PORT = int(sys.argv[2])

#The socket is created
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

except socket.error as e:
    print("Error creating a socket !")
    sys.exit(1);


#The socket seeks to connect to the host
try:
    s.connect((HOST, PORT))

except socket.error as e:
    print("Error connecting to Host !")
    sys.exit(1)


message = input("Enter message to send : ")
try:
    s.sendall(message.encode()) #Message is sent sent to the host

except socket.error as e:
    print("Error sending message to Host !")
    sys.exit(1);

try:
    reply = s.recv(1024) #Message is echoed back 

except socket.error as e:
    print("Error receiving message from Host !")
    sys.exit(1) 

print("Reply from server : ", reply.decode())

try:
    s.close()

except socket.error as e:
    print("Error closing socket !")
    sys.exit(1)