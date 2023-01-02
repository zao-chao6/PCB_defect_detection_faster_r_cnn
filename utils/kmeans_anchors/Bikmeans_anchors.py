import numpy as np
from read_voc import VOCDataSet

# bik-means算法
"""
Args:
    boxes: 需要聚类的bboxes
    k: 簇数(聚成几类)
    dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
"""

def load_data_set(fileName):
    """加载数据集"""
    dataSet = []  # 初始化一个空列表
    fr = open(fileName)
    for line in fr.readlines():
        # 按tab分割字段，将每行元素分割为list的元素
        curLine = line.strip().split('\t')
        # 用list函数把map函数返回的迭代器遍历展开成一个列表
        # 其中map(float, curLine)表示把列表的每个值用float函数转成float型，并返回迭代器
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet


def distance_euclidean(vector1, vector2):
    """计算欧氏距离"""
    return np.sqrt(sum(np.power(vector1-vector2, 2)))  # 返回两个向量的距离


def rand_center(dataSet, k):
    """构建一个包含K个随机质心的集合"""
    n = np.shape(dataSet)[1]  # 获取样本特征值

    # 初始化质心，创建(k,n)个以0填充的矩阵
    centroids = np.mat(np.zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
    # 遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataSet[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 计算每一列的质心，并将其赋给centroids
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    
    # 返回质心
    return centroids


def k_means(dataSet, k, distMeas=distance_euclidean, creatCent=rand_center):
    """K-means聚类算法"""
    m = np.shape(dataSet)[0]  # 行数
    # 建立簇分配结果矩阵，第一列存放该数据所属中心点，第二列是该数据到中心点的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = creatCent(dataSet, k)  # 质心，即聚类点
    # 用来判定聚类是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据划分到离他最近的中心点
            minDist = np.inf  # 无穷大
            minIndex = -1  # 初始化
            for j in range(k):
                # 计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    # 如果第i个数据点到第j中心点更近，则将i归属为j
                    minDist = distJI
                    minIndex = j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):  # 重新计算中心点
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataMat, k, distMeas=distance_euclidean):
    """二分k-means算法"""
    m = np.shape(dataMat)[0]
    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 根据数据集均值获取第一个质心
    centroid0 = np.mean(dataMat, axis=0).tolist()[0]
    # 用一个列表来保留所有的质心
    centList = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的距离
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataMat[j, :]) ** 2
    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while (len(centList) < k):
        # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        lowestSSE = np.inf  # 无穷大
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            # 对每一个簇,将该簇中的所有点看成一个小的数据集
            ptsInCurrCluster = dataMat[np.nonzero(
                clusterAssment[:, 0].A == i)[0], :]
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = k_means(ptsInCurrCluster, 2, distMeas)
            # 划分数据的SSE与未划分的之和作为本次划分的总误差
            sseSplit = sum(splitClustAss[:, 1])  # 划分数据集的SSE
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 未划分数据集的SSE
            print('划分数据集的SSE, and 未划分的SSE: ', sseSplit, sseNotSplit)
            # 将划分与未划分的SSE求和与最小SSE相比较 确定是否划分
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i  # 当前最适合做划分的中心点
                bestNewCents = centroidMat  # 划分后的两个新中心点
                bestClustAss = splitClustAss.copy()  # 划分点的聚类信息
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 更新为最佳质心
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('本次最适合划分的质心: ', bestCentToSplit)
        print('被划分数据集样本数量: ', len(bestClustAss))
        # 更新质心列表
        # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加bestNewCents的第二个质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return np.mat(centList), clusterAssment

def main(img_size=600, k=9, thr=0.25, gen=1000):
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    dataset = VOCDataSet(voc_root="/data", year="2012", txt_name="train.txt")
    im_wh, boxes_wh = dataset.get_info()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # nine_anchors = generate_anchor_base()
    # print(nine_anchors)

    # height, width, feat_stride  = 38,38,16
    # anchors_all                 = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    # print(np.shape(anchors_all))
    
    # fig     = plt.figure()
    # ax      = fig.add_subplot(111)
    # plt.ylim(-300,900)
    # plt.xlim(-300,900)
    # shift_x = np.arange(0, width * feat_stride, feat_stride)
    # shift_y = np.arange(0, height * feat_stride, feat_stride)
    # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # plt.scatter(shift_x,shift_y)
    # box_widths  = anchors_all[:,2]-anchors_all[:,0]
    # box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    # for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
    #     rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
    #     ax.add_patch(rect)
    # plt.show()
    
    # 测试biKmeans算法
    datMat = np.mat(load_data_set(r'F:\Desktop\PCB_code\PCB_DataSet\trainval.txt'))
    # 5个anchor框，
    centList, clusterAssment = biKmeans(datMat, 5)
    print("质心结果：", centList)
    print("聚类结果：", clusterAssment)
    # 可视化
    plt.scatter(np.array(datMat)[:, 0], np.array(datMat)[:, 1], c=np.array(clusterAssment)[:, 0].T)
    plt.scatter(centList[:, 0].tolist(), centList[:, 1].tolist(), c="r")
    plt.show()

