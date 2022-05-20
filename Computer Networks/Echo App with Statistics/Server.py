import socket
import sys
import time

#Port number is provided as command line argument
PORT = int(sys.argv[1])
#Keeps track of socket status.
SocketActive = True

#Function to send messages to Clients
def SendMessage(Message):
    try:
        conn.sendall(Message.encode())

    except socket.error as e:
        print("Error sending message to Client !")

#Function to receive messages from Clients
def RecvMessage():
    try:
        Message = conn.recv(1024).decode()

    except socket.error as e:
        print("Error receiving messages from Client !")
  
    return Message

def ConnectionSetupPhase(SetupMessage):
    OkSetup = "200 OK:Ready"
    Parameters = SetupMessage[:-1].split(" ")
    
    #Check if Setup message is valid
    if(SetupMessage[-1] == "\n" and Parameters[0] == "s" and (Parameters[1], Parameters[2], Parameters[3], Parameters[4] != None )):
        #Asking the client to proceed
        SendMessage(OkSetup)
        print("Setup Success !")
        return Parameters
    else:
        print("404 Error : Invalid Connection Setup Message")
        conn.close()

def MeasurementPhase(Probes, ServerDelay):
    #Receive Probe from Client
    for i in range(1,(Probes + 1)):
        MeasurementMessage = RecvMessage()
        Parameters = MeasurementMessage[:-1].split(" ")
        print("Probe ", Parameters[1] + " received !")

        #Check if Measurement message is valid
        if (int(Parameters[1]) != i or Parameters[0] != "m"):
            print("404 Error : Invalid Measurement Message !")
            conn.close()

        #Serve Delay if specified
        time.sleep(ServerDelay)

        #Echo message back to Client
        SendMessage(MeasurementMessage)

def ConnectionTerminationPhase():
    OkTerminate = "200 OK : Closing Connection"
    TerminateMessage = RecvMessage()

    #Check if Termination message is valid
    if(TerminateMessage[-1] == "\n" and TerminateMessage[0] == "t"):
        #Asking the Client to terminate gracefully
        SendMessage(OkTerminate)
        print("Termination Success !")
        conn.close()
    else:
        print("404 Error : Invalid Connection Termination Message")
        conn.close()


#Socket creation
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

except socket.error as e:
    print("Error creating socket !")
    sys.exit(1)

#Socket is binded to port
try:
    s.bind(('', PORT))

except:
    print("Error binding socket to port !")
    sys.exit(1)

#Socket is put in listening state
#Backlog is set to 5
try:
    s.listen(5)

except socket.error as e:
    print("Error putting socket in listening mode !")

print("The Server is ready to receive !")

while SocketActive:
    #Accepting connections from Clients
    try:
        conn, addr = s.accept()

    except socket.error as e:
        print("Error connecting to Client !")
        sys.exit(1)
    
    print("Connected by : ", addr)

    #Setup Message is received from Client
    Message = RecvMessage()
    
    Parameters = ConnectionSetupPhase(Message)
    
    #Parameters are extracted from the message
    Phase = Parameters[0]
    Measurement = Parameters[1]
    Probes = int(Parameters[2])
    Size = int(Parameters[3])
    ServerDelay = int(Parameters[4])
    
    #Measurement Phase begins
    MeasurementPhase(Probes, ServerDelay)
    
    #Termination Phase begins
    ConnectionTerminationPhase()