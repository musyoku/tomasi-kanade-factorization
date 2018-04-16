import numpy as np
from chainer import functions as fn, Variable


def recover_3d_structure(registered_measurement_matrix):
    num_frames = registered_measurement_matrix.shape[0] // 2
    U, Z, V = np.linalg.svd(registered_measurement_matrix, full_matrices=False)
    Z = np.diag(Z)
    R_ = np.dot(U[:, :3], np.sqrt(Z[:3, :3]))
    S_ = np.dot(np.sqrt(Z[:3, :3]), V[:3])
    I = R_[:num_frames]
    J = R_[num_frames:]
    Q = Variable(np.eye(3, dtype=np.float32) + np.random.normal(0, 0.1, (3, 3)).astype(np.float32))

    lr = 0.1
    minimum_loss_value = 0.0001
    target_ii = np.full((num_frames, ), 1.0, dtype=np.float32)
    target_jj = np.full((num_frames, ), 1.0, dtype=np.float32)
    target_ij = np.full((num_frames, ), 0.0, dtype=np.float32)
    for itr in range(1000):
        loss_ii = fn.sum(fn.matmul(fn.matmul(I, Q), Q.T) * I, axis=1)
        loss_jj = fn.sum(fn.matmul(fn.matmul(J, Q), Q.T) * J, axis=1)
        loss_ij = fn.sum(fn.matmul(fn.matmul(I, Q), Q.T) * J, axis=1)
        loss = fn.mean_squared_error(loss_ii, target_ii) + fn.mean_squared_error(
            loss_jj, target_jj) + fn.mean_squared_error(loss_ij, target_ij)
        Q.cleargrad()
        loss.backward()
        Q.data -= lr * Q.grad
        if float(loss.data) < minimum_loss_value:
            break
        lr *= 0.995

    R = np.dot(R_, Q.data)
    S = np.dot(np.linalg.inv(Q.data), S_)
    return R, S, R_, S_