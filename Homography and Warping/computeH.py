import numpy as np

def computeH(t1, t2):
    #Compute p1
    p1 = np.append(t1, np.ones((1, t1.shape[1])), axis=0)

    #Create A array
    A = np.array([
        np.row_stack((p1[:, 0].T, np.zeros(3), - t2[0, 0] * p1[:, 0].T)).flatten(),
        np.row_stack((np.zeros(3), p1[:, 0].T, - t2[1, 0] * p1[:, 0].T)).flatten(),
    ])
    for i in range(1, t1.shape[1]):
        A = np.row_stack((A, np.row_stack((p1[:, i].T, np.zeros(3), - t2[0, i] * p1[:, i].T)).flatten()))
        A = np.row_stack((A, np.row_stack((np.zeros(3), p1[:, i].T, - t2[1, i] * p1[:, i].T)).flatten()))

    #Find least eigen value
    _, _, vt = np.linalg.svd(A)
    least_eigen_valued_vector = vt[-1]
    return least_eigen_valued_vector.reshape((3, 3))

