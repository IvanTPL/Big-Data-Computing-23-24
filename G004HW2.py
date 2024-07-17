from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from pyspark.sql import SparkSession
import time


def calculate_rs(lst_cells):
    d = {}
    for i in range(len(lst_cells)):
        if d.get(lst_cells[i][0]) is None:
            d[lst_cells[i][0]] = [lst_cells[i][1], lst_cells[i][1], lst_cells[i][1]]
        for j in range(i+1, len(lst_cells)):
            if d.get(lst_cells[j][0]) is None:
                d[lst_cells[j][0]] = [lst_cells[j][1], lst_cells[j][1], lst_cells[j][1]]
            if (lst_cells[j][0][0] >= lst_cells[i][0][0] - 3) & (lst_cells[j][0][0] <= lst_cells[i][0][0] + 3) & (lst_cells[j][0][1] >= lst_cells[i][0][1] - 3) & (lst_cells[j][0][1] <= lst_cells[i][0][1] + 3):
                d[lst_cells[i][0]][1] += lst_cells[j][1]
                d[lst_cells[j][0]][1] += lst_cells[i][1]
            if (lst_cells[j][0][0] >= lst_cells[i][0][0] - 1) & (lst_cells[j][0][0] <= lst_cells[i][0][0] + 1) & (lst_cells[j][0][1] >= lst_cells[i][0][1] - 1) & (lst_cells[j][0][1] <= lst_cells[i][0][1] + 1):
                d[lst_cells[i][0]][0] += lst_cells[j][1]
                d[lst_cells[j][0]][0] += lst_cells[i][1]
    return [(k, d[k]) for k in d.keys()]

def points_count(lst, D):
    cells_dict = {}
    for point in lst:
        i = (point[0] * 2**1.5) // D
        j = (point[1] * 2**1.5) // D
        if (i, j) not in cells_dict.keys():
            cells_dict[(i, j)] = 1
        else:
            cells_dict[(i, j)] += 1
    return [(key, cells_dict[key]) for key in cells_dict.keys()]

def MRApproxOutliers(docs, D, M):
    res = (docs.mapPartitions(lambda x: points_count(x, D)).reduceByKey(lambda x, y: x + y))
    res_t = calculate_rs(res.collect())
    sure = [] 
    unsure = []
    num_sure = 0
    num_unsure = 0
    for x in res_t:
        if x[1][1] <= M:
            sure.append(x[0])
            num_sure += x[1][2]
        elif x[1][0] <= M:
            unsure.append(x[0])
            num_unsure += x[1][2]
    print("Number of sure outliers = {}".format(num_sure))
    print("Number of uncertain points = {}".format(num_unsure))

def ExactOutliers(lst, D, M, K):
    d = {}
    for i in range(len(lst)):
        d[(lst[i][0], lst[i][1], i)] = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            d_1 = lst[j][0] - lst[i][0]
            d_2 = lst[j][1] - lst[i][1]
            dist = d_1 * d_1 + d_2 * d_2
            if dist <= D * D:
                d[(lst[i][0], lst[i][1], i)] += 1
                d[(lst[j][0], lst[j][1], j)] += 1
    res = sorted(d.items(), key = lambda x: x[1])
    outliers = []
    for x in res:
        if x[1] > M:
            break
        else:
            outliers.append((x[0][0],x[0][1]))
    print("Number of Outliers = {}".format(len(outliers)))
    for i in range(min(K, len(outliers))):
        print("Point: {}".format(outliers[i]))


def SequentialFFT(P, K):
    ind = rand.randint(0, len(P)-1)
    centers = [ind] # array of indices of center points
    cnt = 1
    dists = [-1 for p in P] # array of distances from each point to set of centers
    last = P[ind] # last assigned center
    while cnt < K:
        maxdist = 0
        pos = -1
        for i in range(len(P)):
            if i in centers:
                continue
            d1 = P[i][0] - last[0] 
            d2 = P[i][1] - last[1]
            d = d1*d1 + d2*d2
            if (dists[i] == -1) | (d < dists[i]):
                dists[i] = d
            if dists[i] > maxdist:
                maxdist = dists[i]
                pos = i
        centers.append(pos)
        last = P[pos]
        cnt+=1
    C = [P[i] for i in centers]
    return C

def radius(points, centers):
    maxdist = 0
    for p in points:
        mind = -1
        for c in centers:
            d1 = p[0] - c[0]
            d2 = p[1] - c[1]
            d = d1*d1 + d2*d2
            if (mind == -1) | (d < mind):
                mind = d
        if mind > maxdist:
            maxdist = mind
    return [maxdist**0.5]
            
def MRFFT(P, K):
    start1 = time.time()
    res = (P.mapPartitions(lambda x: SequentialFFT(list(x), K))).cache()
    res.count()
    end1 = time.time()

    start2 = time.time()
    C = SequentialFFT(res.collect(), K)
    end2 = time.time()

    C_shared = sc.broadcast(C)
    start3 = time.time()
    R = (P.mapPartitions(lambda x: radius(x, C_shared.value)).reduce(lambda x, y: max(x, y)))
    end3 = time.time()

    print("Running time of MRFFT Round 1 = {} ms".format(int(round(end1 - start1, 3) * 1000)))
    print("Running time of MRFFT Round 2 = {} ms".format(int(round(end2 - start2, 3) * 1000)))
    print("Running time of MRFFT Round 3 = {} ms".format(int(round(end3 - start3, 3) * 1000)))
    return R


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

   
def main():
    path = sys.argv[1]
    M = int(sys.argv[2])
    K = int(sys.argv[3])
    L = int(sys.argv[4])
    print("{} M={} K={} L={}".format(path, M, K, L))
    
    conf = SparkConf().setAppName("G004HW2")
    conf.set("spark.locality.wait", "0s")
    rawData = spark.read.csv(path).rdd
    inputPoints = rawData.map(lambda x: [float(s) for s in x]).repartition(numPartitions=L).cache()
    print("Number of points = {}".format(inputPoints.count()))
    
    D = MRFFT(inputPoints, K)
    print("Radius = {}".format(D))
    
    start = time.time()
    MRApproxOutliers(inputPoints, D, M)
    end = time.time()
    print("Running time of MRApproxOutliers = {} ms".format(int(round(end - start, 3) * 1000)))
    

if __name__ == "__main__":
	main()