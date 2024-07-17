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

def MRApproxOutliers(docs, D, M, K):
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
    out = res.sortBy(keyfunc = lambda x: x[1]).take(K)
    print("Number of sure outliers = {}".format(num_sure))
    print("Number of uncertain points = {}".format(num_unsure))
    for x in out:
        print("Cell: {}  Size = {}".format((int(x[0][0]), int(x[0][1])), x[1]))

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
        
def main():
    path = sys.argv[1]
    D = float(sys.argv[2])
    M = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])
    print("{} D={} M={} K={} L={}".format(path, D, M, K, L))
    
    conf = SparkConf().setAppName("G004HW1")
    spark = SparkSession.builder.getOrCreate()
    rawData = spark.read.csv(path).rdd
    inputPoints = rawData.map(lambda x: [float(s) for s in x]).repartition(numPartitions=L).cache()
    print("Number of points = {}".format(inputPoints.count()))
    
    if inputPoints.count() <= 200000:
        listOfPoints = inputPoints.collect()
        start = time.time()
        ExactOutliers(listOfPoints, D, M, K)
        end = time.time()
        print("Running time of ExactOutliers = {} ms".format(int(round(end - start, 3) * 1000)))
    
    start = time.time()
    MRApproxOutliers(inputPoints, D, M, K)
    end = time.time()
    print("Running time of MRApproxOutliers = {} ms".format(int(round(end - start, 3)*1000)))
    

if __name__ == "__main__":
	main()