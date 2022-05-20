import socket
import sys

#Port Number is provided as command line argument 
PORT = int(sys.argv[1])
#State of the socket : open/close
SocketActive = True

#Socket is created and binded to the given port
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

except socket.error as e:
    print("Error creating socket !")
    sys.exit(1)

try:
    s.bind(('', PORT))

except:
    print("Error binding socket to port !")
    sys.exit(1)

#The socket is put in listening state
#Backlogs is set to 5
try:
    s.listen(5)

except socket.error as e:
    print("Error putting socket in listening mode !")
    sys.exit(1)

print("The server is ready to receive !")

while SocketActive:
    try:
        conn, addr = s.accept() #Accepts connection from the client

    except socket.error as e:
        print("Error accepting connection !")
        sys.exit(1)
    
    print("Connected by : ", addr)

    with conn :
        try:
            message = conn.recv(1024) #Receive message from the client
        
        except socket.error as e:
            print("Error receiving message from Client !")
            sys.exit(1)
        
        try:
            conn.sendall(message) #Echoes the message back

        except socket.error as e:
            print("Error sending message to Client !")
            sys.exit(1)           

    if(message.decode() == "End"): #Closes the socket if 'End' is received
        SocketActive = False
try:
    s.close()

except socket.error as e:
    print("Error closing socket !")
    sys.exit(1)