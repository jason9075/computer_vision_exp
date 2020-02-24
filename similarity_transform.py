from skimage import transform as trans
import numpy as np

SRC = np.array([
    [38.2, 51.6],
    [73.5, 51.5],
    [56.0, 71.7],
    [41.5, 92.3],
    [70.7, 92.2]], dtype=np.float32)
DST = np.array([
    [38.4, 51.7],
    [73.8, 51.2],
    [56.2, 71.4],
    [41.7, 92.1],
    [70.1, 92.9]], dtype=np.float32)


def by_api():
    tform = trans.SimilarityTransform()
    tform.estimate(SRC, DST)
    M = tform.params
    print(M)


def by_custom():
    # paper "Least-squares estimation of transformation parameters between two point patterns"
    estimate_scale = True

    num = SRC.shape[0]
    dim = SRC.shape[1]

    # Compute mean of src and dst.
    src_mean = SRC.mean(axis=0)
    dst_mean = DST.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = SRC - src_mean
    dst_demean = DST - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    print(T)


if __name__ == '__main__':
    by_api()
    by_custom()
