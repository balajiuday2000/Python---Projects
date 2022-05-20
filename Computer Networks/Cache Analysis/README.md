# Web Caching - GENI Mini Project

### CS655 - Graduate Introduction To Computer Networks
### Group members:
- Balaji Udayakumar (balaji@bu.edu)
- Duruvan Saravanan (durusarv@bu.edu)
- Luke Staib (ljstaib@bu.edu)
- Syahrial Bin Dahler (dahler@bu.edu)
 
# 1. INTRODUCTION

<p>This experiment aims to demonstrate the benefits of caching. In the first part, we   compare web retrieval performance with and without caching. In the second part, we compare web retrieval performances of different caches implementing different caching algorithms. Various metrics such as hit rate, miss rate, throughput, and delay are used to measure performance. Further analysis is done by modifying loss rate.</p>

# 2. EXPERIMENTAL METHODOLOGY

## 2.1 (PART 1) Web retrieval performance with and without caching

### 2.1.1. DESIGN

<p>The Client and LRU Cache server (or No cache server) are set up on GENI. With the LRU cache, the client has been configured to access the cache first to retrieve a web object and if a miss occurs, the Cache server accesses the origin Server, caches the web object and provides it to the Client. The LRU Cache has been implemented as a doubly linked list.</p>

### 2.1.2. METHODOLOGY

#### 2.1.2.1. SET UP

<p>Set up the topology as shown in the design above and load the configuration files (`client.py`, `LRUServer.py`, or `server_nocache.py`) into the respective machines.</p>

#### 2.1.2.2. EXPERIMENT

- Run `client.py` on the Client and `LRUserver.py` on the Server. Have the scripts run in their entirety. At the end, the script will output average throughput, average RTT, and average memory access time for the Client.
- Run `client.py` on the Client and `server_nocache.py` on the Server. Have the scripts run in their entirety. At the end, the script will output average throughput,average RTT, and average memory access time for the Client.
- Run `nginx_client.py` on the same node that has nginx installed. Results are produced in the same way as above.
- Compare their performances.

## 2.2 (PART 2) Comparing Caches

### 2.2.1. DESIGN

<p>The Client and different cache servers are set up on GENI. The client has been configured to access the cache first to retrieve a web object and if a miss occurs, the Cache server accesses the origin Server, caches the web object and provides it to the Client.The FIFO cache has also been implemented using a doubly linked list.</p>

### 2.2.2 METHODOLOGY

#### 2.1.2.1. SET UP

<p>Similarly to part 1, the topology is set up and the configuration files (`client.py` (or `nginx_client.py`), `LRUServer.py`(or `FIFOServer.py`)) are loaded into the respective machines.</p>

#### 2.1.2.2. EXPERIMENT

- Run `client.py` on the Client and `LRUserver.py` or (`FIFOServer.py` or the nginx Server)  on the Server. Have the scripts run in their entirety. At the end, the script will output average throughput,average RTT, and average memory access time for the Client.
- Repeat the experiment by varying the loss values.
- Compare their performances.

# 3. USAGE INSTRUCTIONS

1. Setting up the environment
	-Reserve a slice with 2 nodes
	- Install python on both the nodes
	- One node acts as the server, another as the client
	- Load the files, LRUCache_server.py, server_nocache.py, FIFOServer.py and requirements.txt into the server
	- Load client.py and source.csv into the client

2. Set up the virtual environments on both machines (client and server) using the following commands:

`pip install virtualenv`
`virtualenv env`

3. Activate the virtual environment on both machines
`source env\bin\activate`

- Install requirements from requirements.txt
`pip install -r requirements.txt`

4. Setup nginx

- Navigate to the cache directory:
`cd CS-655-caching-project-main/cache`

- Execute our bash script to install:
`bash nginx-install.sh`

The bash script contains instructions to install nginx and then copies the configuration file from the downloaded github repo to the correct location and starts nginx. It also installs the necessary python packages. 

- To run the experiment:
`sudo python nginx_client.py`

5. To run Part 1 (Web retrieval performance with and without caching):

Without caching:
- At server: 
`python server_nocache.py`
- At client:
`python client.py [ip_address_of_server] [cache_type] [loss] [delay]`

With our LRU Cache :
- At server:
`python LRUCache_Server.py`
- At client:
`python client.py [ip_address_of_server] [cache_type] [loss] [delay]`

6. To run Part 2 (Comparing caches):

With our LRU Cache: 
- Same as before. See the previous step.

With our FIFO Cache:
- At server:
`python FIFOServer.py`
- At client:
`python client.py [ip_address_of_server] [cache_typ] [loss] [delay]`

7. To run nginx client:
- On the same node that has nginx installed: 
`sudo python nginx_client.py`
