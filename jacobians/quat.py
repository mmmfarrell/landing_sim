from __future__ import print_function

import numpy as np

def skew(vec):
    skew = np.zeros((3, 3))
    skew[0, 1] = -vec[2]
    skew[0, 2] = vec[1]

    skew[1, 0] = vec[2]
    skew[1, 2] = -vec[0]

    skew[2, 0] = -vec[1]
    skew[2, 1] = vec[0]

    return skew

class Quat():
    def __init__(self):
        self.arr = np.zeros((4,))
        self.arr[0] = 1.

    def __str__(self):
        return str(self.arr[0]) + ", " + str(self.arr[1]) + " i, " \
                + str(self.arr[2]) + " j, " + str(self.arr[3]) + " k"

    def set_identity(self):
        self.arr = np.zeros((4,))
        self.arr[0] = 1.

    def normalize(self):
        self.arr /= np.linalg.norm(self.arr)

    def random(self):
        self.arr = np.random.rand(4)
        self.normalize()

    def otimes(self, qa, qb):
        qa0 = qa[0]
        qa1 = qa[1]
        qa2 = qa[2]
        qa3 = qa[3]
        qabar = qa[1:]

        qb0 = qb[0]
        qb1 = qb[1]
        qb2 = qb[2]
        qb3 = qb[3]
        qbbar = qb[1:]

        result = np.zeros((4, 1))
        result[0] = qb0 * qa0 - qb1 * qa1 - qb2 * qb2 - qb3 * qa3
        result[1] = qb0 * qa1 + qb1 * qa0 - qb2 * qb3 + qb3 * qa2
        result[2] = qb0 * qa2 + qb1 * qa3 + qb2 * qb0 - qb3 * qa1
        result[3] = qb0 * qa3 - qb1 * qa2 + qb2 * qb1 + qb3 * qa0

        result /= np.linalg.norm(result)
        return result

    def boxplus(self, delta):
        qtilde = self.exp(delta)
        self.arr = self.otimes(self.arr, qtilde)

        return self.arr

    def exp(self, delta):
        q_tilde = np.zeros((4, 1))
        q_tilde[0] = 1.
        q_tilde[1:] = 0.5 * delta
        q_tilde /= np.linalg.norm(q_tilde)
        return q_tilde

    def log(self):
        return 2. * np.sign(q[0]) * q[1:4]

if __name__ == '__main__':
    q = Quat()

    print('q: ', q)
    #  q.random()
    #  print('q: ', q)

    dx = np.zeros((3, 1))
    q.boxplus(dx)
    print('q: ', q)

    dx = 1e-3 * np.random.rand(3, 1)
    q.boxplus(dx)
    print('q: ', q)

    print('done')
