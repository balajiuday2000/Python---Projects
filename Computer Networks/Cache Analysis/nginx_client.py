import time
import os
import pandas as pd
import random
import subprocess

hitCount = 0
missCount = 0


def set_up_experiment():
    global hitCount
    global missCount
    data = pd.read_csv("source.csv")
    # print(data.loc[0])

    # return
    source_data = data["website"].tolist()

    iteration_num = 20  # number of iteration for each probe
    experiment_num = 100
    miss = []
    hit = []
    loss = []

    # initiate empty array to store delay or throughput
    average_rtt = []
    average_tput = []

    for _ in range(experiment_num):
        rtt = []
        tput = []
        hitCount = 0
        missCount = 0
        loss_count = 0

        for i in range(iteration_num):
            # generate random number for loss
            loss_chance = random.random()
            if loss_chance > 0:
                # request is not lost

                # choose a random website to make the request to
                req = random.choice(source_data)
                print("iteration ", i)
                print("curl -x localhost:8888 " + req)

                # send the request
                start_time = time.time()
                subprocess.call("curl -o test.txt -x localhost:8888 " + req, shell=True)
                end_time = time.time()
                t = end_time - start_time

                # collect throughput and rtt
                tput.append(os.path.getsize('test.txt') / t)
                rtt.append(t)
            else:
                # lose the request
                print("REQUEST LOSS")
                loss_count += 1
                tput.append(0)
                rtt.append(60)

        # get hit and miss counts from the custom log file
        counts = read_nginx_log()

        # calculate average rtt and tput for the 20 requests
        average_rtt.append(sum(rtt) / iteration_num)
        average_tput.append(sum(tput) / iteration_num)

        # record cache miss, hit and loss
        miss.append(counts[1] - missCount)
        loss.append(loss_count)
        hit.append(counts[0] - hitCount)
        hitCount += counts[0]
        missCount += counts[1]

    print("Miss total ", sum(miss) / experiment_num)
    print("Hit total ", sum(hit) / experiment_num)

    # save data to a csv file
    rtt_df = pd.DataFrame()
    rtt_df["RTT"] = average_rtt
    rtt_df["MISS"] = miss
    rtt_df["HIT"] = hit
    rtt_df["LOSS"] = loss
    rtt_df["THROUGHPUT"] = average_tput

    rtt_df.to_csv('rtt.csv')


def read_nginx_log():
    global hitCount
    global missCount
    hit = 0
    miss = 0

    # log file location
    file = '/var/log/nginx/cache_access.log'
    f = open(file, "r")
    for line in f:
        print(line)
        if "HIT" in line:
            # count cache hit
            hit += 1
        else:
            # count cache miss
            miss += 1

    return [hit, miss]


if __name__ == '__main__':
    set_up_experiment()
