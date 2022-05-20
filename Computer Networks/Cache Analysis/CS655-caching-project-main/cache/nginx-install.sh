#!/bin/bash

#install nginx
sudo apt update
sudo apt install nginx

#stop nginx
sudo nginx -s stop

#copy nginx config file
sudo cp -a nginx.conf /etc/nginx/nginx.conf

#restart nginx
sudo nginx
sudo nginx -s reload

#install python requirements
sudo apt install python-pip
pip install pandas