import socket
import sys
import time

#Host address and port number are provided as command line arguments
HOST = sys.argv[1]
PORT = int(sys.argv[2])

#Function to send messages to Host
def SendMessage(Message):
    try:
        s.sendall(Message.encode())

    except socket.error as e:
        print("Error sending message to Host !")

#Function to receive messages from Host
def RecvMessage():
    try:
        Message = s.recv(1024).decode()

    except socket.error as e:
        print("Error receiving messages from Host !")
  
    return Message

def ConnectionSetupPhase(Measurement, Size, Probes, ServerDelay):
    #The setup message is put together with the input from User
    SetupMessage = "s" + " " + Measurement + " " + Probes + " " + Size + " " + ServerDelay + "\n"
    SendMessage(SetupMessage)

#To calculate size of the probe in bytes
def utf8len(s):
    return len(s.encode('utf-8'))

def MeasurementPhase(Probes, Size, Measurement):
    RTTs = []
    Tputs = []
    for i in range(1,(int(Probes) +1)):
        #The probe is put together
        ProbeMessage = "m" + " " + str(i) + " " + str(bytes(Size, encoding='utf-8')) + "\n"
        Length1 = utf8len(ProbeMessage)
        SendMessage(ProbeMessage)

        #The time at which the probe is sent is calculated
        SentAt = time.time()
        Echo = RecvMessage()
        #The time at which the echo is received is calculated
        ReceivedAt = time.time()
        
        print("Probe ", i, " echoed back !")
        Length2 = utf8len(Echo)

        #Measurement is calculated only if entire probe if echoed back
        if(Length1 == Length2):
            print("The entire probe is echoed back ! \n")
            if(Measurement == "rtt"):
                RTTs.append((ReceivedAt - SentAt))
    
            elif(Measurement == "tput"):
                Tputs.append((int(Size) / (ReceivedAt - SentAt)))
    
    
        else:
            print("Parts of echo missing !")

    if(Measurement == "rtt"): return RTTs
    else : return Tputs
    

    
    

def ConnectionTerminationPhase():
    #The Termination message is put together
    TerminateMessage = "t" + "\n"
    SendMessage(TerminateMessage)

#Socket is created 
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

except socket.error as e:
    print("Error creating scoket !")
    sys.exit(1)

#Socket seeks to connect to Host
try:
    s.connect((HOST, PORT))

except:
    print("Error connecting to Host !")
    sys.exit(1)


#Connection Setup Phase
OkSetup = "200 OK:Ready"
Measurement = input(" Enter 'rtt' if you want to measure the roundtrip time or 'tput' if you want to measure the throughput time : ")
Probes = input("Enter the number of probes to be sent : ")
Size = input("Enter the message size in bytes : ")
ServerDelay = input("Enter the server delay : ")

ConnectionSetupPhase(Measurement, Size, Probes, ServerDelay)

if(RecvMessage() == OkSetup):
    print("Server has accepted the request !")
    print("Sending probes now !")



if(Measurement == "rtt"):
    RTTs = MeasurementPhase(Probes, Size, Measurement)
    #Mean RTT is calulated
    MeanRTT = sum(RTTs) / int(Probes) 
    print("Mean RTT = ", MeanRTT, " seconds")
else:
    Tputs = MeasurementPhase(Probes, Size, Measurement)
    #Mean Throughput is calulated
    MeanTput = sum(Tputs) / int(Probes)
    print(" Mean Throughput = ", MeanTput, " bytes/second" )


#The connection is terminated gracefully
OkTerminate = "200 0K : Closing Connection"
ConnectionTerminationPhase()
if(RecvMessage() == OkTerminate):
    try:
        s.close()

    except socket.error as e:
        print("Error terminating connection gracefully !")


    