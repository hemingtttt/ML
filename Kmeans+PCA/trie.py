import mat73
import numpy as np
import scipy

import mindspore as ms
import mindspore.ops as ops

import tqdm


def data_group_avg(group_ids: ms.Tensor, data: ms.Tensor, k: int = 4):
    """计算每个簇的聚类中心
    Args:
        group_ids (ms.Tensor): shape=(num_pts,)
        data (ms.Tensor): shape=(num_pts, 128)
        k (int): 划分成的簇的数量
    Returns:
        avg_by_group (ms.Tensor): shape=(k, 128)
    """
    # 参见 https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unsorted_segment_sum.html
    # 用法：unsorted_segment_sum(input_x, segment_ids, num_segments)
    unsorted_segment_sum = ops.UnsortedSegmentSum()
    ones_like = ops.OnesLike()

    # TODO: 计算新的聚类中心，建议使用 unsorted_segment_sum，或可自行编写代码求解
    # sum_by_group: 每簇的数据点向量和
    sum_by_group = unsorted_segment_sum(data, group_ids, k)
    # count_by_group: 每簇的数据点个数
    ones = ones_like(group_ids)
    count_by_group = unsorted_segment_sum(ones, group_ids, k).reshape((k, 1))
    avg_by_group = sum_by_group / count_by_group

    return avg_by_group


def calc_clusters(data_points: ms.Tensor, centroids: ms.Tensor, k: int) -> ms.Tensor:
    """计算每个点与聚类中心之间的距离，并重新划分聚类
    Args:
        data_points (Tensor): shape=(num_pts, num_feats)
        centroids (Tensor): shape=(num_clusters, num_feats)
            current cluster centers
        k (int): num_clusters

    Returns:
        centroid_group (ms.Tensor): shape=(num_pts,)
    """
    reshape = ops.Reshape()
    reduce_sum = ops.ReduceSum(keep_dims=False)
    square = ops.Square()
    argmin = ops.Argmin()

    num_pts, num_feats = data_points.shape
    # tile centroids to shape (num_pts, k, num_feats)
    centroid_matrix = ops.repeat_elements(
        reshape(centroids, (1, k, num_feats)), num_pts, axis=0
    )
    # TODO: 计算每个点与聚类中心之间的距离，并重新划分簇
    # 计算平方欧氏距离
    expanded = reshape(data_points, (num_pts, 1, num_feats))
    diff = expanded - centroid_matrix
    dists = reduce_sum(square(diff), axis=2)
    # 分配时，以每个数据点最小距离为最接近的中心点
    centroid_group = argmin(dists)

    return centroid_group


def k_means(data_x: ms.Tensor, k: int, iterations: int):
    """K-Means 聚类主流程"""
    num_pts, num_feats = data_x.shape

    # 随机初始化中心
    rand_starts = np.array([data_x[np.random.choice(num_pts)] for _ in range(k)])
    centroids = ms.Tensor(rand_starts.astype(np.float32))
    data_points = ms.Tensor.from_numpy(data_x.astype(np.float32))

    for _ in tqdm.trange(iterations, desc="K-Means", ncols=80):
        # 步骤1: 分配簇
        groups = calc_clusters(data_points, centroids, k)
        # 步骤2: 更新中心
        centroids = data_group_avg(groups, data_points, k)
    return centroids


def trie_learn(data: ms.Tensor, centers: ms.Tensor):
    """计算三角化嵌入所需参数"""
    # 确保 data 为 Tensor
    if not isinstance(data, ms.Tensor):
        data = ms.Tensor.from_numpy(data)
    n_samples, n_features = data.shape
    k = centers.shape[0]

    x_mean = ops.Zeros()((k * n_features,), dtype=ms.float32)
    for i in range(k):
        # TODO 求取特征均值 x_mean (k * n_features)
        # i_mean = ...
        # x_mean[...] = ...
        # 计算每个簇中心与所有样本的归一化差值和
        diff = data - centers[i]
        normed = diff / ops.norm(diff, dim=1, keepdim=True)
        i_mean = ops.ReduceSum()(normed, axis=0)
        x_mean[i * n_features:(i + 1) * n_features] = i_mean
    # 对所有样本求平均
    x_mean = x_mean / n_samples

    # 求取协方差矩阵 cov_d (k*n_features, k*n_features)
    cov_d = ops.Zeros()((k * n_features, k * n_features), dtype=data.dtype)
    # 根据自己的内存/显存 大小调整 step
    step = 1024
    for i in range(0, n_samples, step):
        end_i = min(i + step, n_samples)
        h_d = ops.Zeros()((end_i - i, n_features * k), dtype=data.dtype)
        for j in range(k):
            h_d_h = data[i:end_i] - centers[j:j+1]
            h_d[:, j * n_features : (j + 1) * n_features] = (
                h_d_h / ops.norm(h_d_h, dim=1, keepdim=True)
            )
        # 去均值并累加
        h_d = h_d - x_mean
        cov_d = cov_d + ops.dot(h_d.T, h_d)
    # 这里依据数学公式应除以 n_samples，但不影响特征排序
    cov_d = cov_d / n_samples

    # 计算特征值和特征向量
    eigval, eigvec = np.linalg.eig(cov_d.asnumpy())
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    return x_mean, eigvec, eigval


def run_trie(kc, iterations):
    """程序起点"""
    # 加载Flickr60k数据
    v_train = scipy.io.loadmat(f"./data/vtrain_{2**19}.mat")["vtrain"].T
    v_train = v_train.astype(np.float32)
    print(f"load Flickr60k train data: {v_train.shape[0]} items.")

    # 使用K-means聚类算法
    c = k_means(v_train, kc, iterations)
    # 保存计算出的聚类中心
    np.savetxt(f"./temp/center_{kc}.csv", c.asnumpy())

    # 计算三角化嵌入所需参数
    x_mean, eigvec, eigval = trie_learn(v_train, c)
    print("trie finish")

    # 计算投影矩阵
    eigval[-128:] = eigval[-129]
    p_emb = np.diag(np.power(eigval, -0.5)) @ eigvec.T

    # 保存
    np.savetxt(f"./temp/x_mean_{kc}.csv", x_mean.asnumpy())
    np.savetxt(f"./temp/eigval_{kc}.csv", eigval)
    np.savetxt(f"./temp/eigvec_{kc}.csv", eigvec)
    np.savetxt(f"./temp/p_emb_{kc}.csv", p_emb)
