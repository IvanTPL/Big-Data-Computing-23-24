from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import math
import random as rand


n = -1

def process_batch(time, batch):
    global streamLength, histogram, msample, m, sticky_hash, r
    batch_size = batch.count()
    if streamLength[0]>=n:
        return
    streamLength[0] += batch_size
    
    if streamLength[0] >= n:
        batch_lst = batch.map(lambda s: int(s)).take(batch_size - (streamLength[0] - n))
        batch_items = {}
        for e in batch_lst:
            if batch_items.get(e) is None:
                batch_items[e] = 1
            else:
                batch_items[e] += 1
    else:
        batch_items = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()
        batch_lst = batch.map(lambda s: int(s)).collect()        
    
    for i in range(len(batch_lst)):
        if len(msample) < m:
            msample.append(batch_lst[i])
        else:
            if rand.random() <= m / (streamLength[0] - batch_size + i + 1):
                pos = rand.randrange(0, m)
                msample[pos] = batch_lst[i]
        if batch_lst[i] in sticky_hash.keys():
            sticky_hash[batch_lst[i]] += 1
        else:
            if rand.random() <= r / n:
                sticky_hash[batch_lst[i]] = 1
    
    for key in batch_items:
        if key not in histogram:
            histogram[key] = batch_items[key]
        else:
            histogram[key] += batch_items[key]
 
    if streamLength[0] >= n:
        stopping_condition.set()



if __name__ == '__main__':
    assert len(sys.argv) == 6

    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")

    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)
    ssc.sparkContext.setLogLevel("ERROR")
    
    stopping_condition = threading.Event()

    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])
    print("INPUT PROPERTIES")
    print("n = {} phi = {} epsilon = {} delta = {} port = {}".format(n, phi, epsilon, delta, portExp))
    streamLength = [0]
    histogram = {}
    msample = []
    m = math.ceil(1 / phi)
    sticky_hash = {}
    r = math.log(1 / (delta * phi)) / epsilon
    
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    #print("Starting streaming engine")
    ssc.start()
    #print("Waiting for shutdown condition")
    stopping_condition.wait()
    #print("Stopping the streaming engine")
    ssc.stop(False, True)
    #print("Streaming engine stopped")

    print("EXACT ALGORITHM")
    print("Number of items in the data structure =", len(histogram))
    res = sorted(histogram.items(), key = lambda x: x[1], reverse=True)
    true_freq = []
    for e in res:
        if (e[1]/n) > phi:
            true_freq.append(e[0])
        else:
            break
    print("Number of true frequent items =", len(true_freq))
    print("True frequent items:")
    for i in sorted(true_freq):
        print(i)
    
    print("RESERVOIR SAMPLING")
    est_freq = set(msample)
    print("Size m of the sample =", m)
    print("Number of estimated frequent items =", len(est_freq))
    print("Estimated frequent items:")
    for e in sorted(list(est_freq)):
        if e in true_freq:
            print("{} +".format(e))
        else:
            print("{} -".format(e))
       
    print("STICKY SAMPLING")
    res_s = sorted(sticky_hash.items(), key = lambda x: x[1], reverse=True)
    sticky_freq = []
    for e in res_s:
        if e[1] >= (phi - epsilon)*n:
            sticky_freq.append(e[0])
        else:
            break
    print("Number of items in the Hash Table =", len(sticky_hash))
    print("Number of estimated frequent items =", len(sticky_freq))
    print("Estimated frequent items:")
    for e in sorted(sticky_freq):
        if e in true_freq:
            print("{} +".format(e))
        else:
            print("{} -".format(e))
