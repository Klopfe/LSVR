import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.datasets import make_regression
from ssvr.classes import SSVR


def primal_nu_SVR_constrained(X, y, cost=1, nu=0.5):

    l, n = X.shape
    cost = cost
# quadratic term matrix
    Q = np.zeros(((2*l+n+2), (2*l+n+2)))
    Q[:n, :n] = np.identity(n)
    Q = matrix(Q)
# Linear term vector

    L = np.zeros((2*l+n+2))
    L[n:(2*l+n+1)] = np.append([np.repeat((cost/l), (2*l))], [(cost*nu)])
    L = matrix(L)
# Matrix of constraints (inequality)
    G = np.zeros(((4*l+n+1), (2*l+n+2)))
    G[:l, :n] = X
    G[:l, n:(2*l+n)] = np.concatenate(
                        (-np.eye(l, l), np.zeros((l, l))), axis=1)
    G[:l, (2*l+n)] = np.repeat((-1), l)
    G[:l, (2*l+n+1)] = np.repeat(1, l)

    G[l:(2*l), :n] = -X
    G[l:(2*l), n:(2*l+n)] = np.concatenate(
                            (np.zeros((l, l)), -np.eye(l, l)), axis=1)
    G[l:(2*l), (2*l+n)] = np.repeat((-1), l)
    G[l:(2*l), (2*l+n+1)] = np.repeat((-1), l)

    G[(2*l):(3*l), n:(l+n)] = -np.eye(l, l)
    G[(3*l):(4*l), (l+n):(2*l+n)] = -np.eye(l, l)

    G[(4*l), (2*l+n)] = -1
    G[(4*l+1):(4*l+n+1), :n] = -np.eye(n, n)
    G = matrix(G)
# Matrix of constraints (equality)
    A = np.repeat(0.0, (2*l+n+2))
    A[:n] = np.repeat(1.0, n)
    A
    A = matrix(A, (1, (2*l+n+2)))

# vector of inequality constraints

    h = np.hstack((y, -y, np.repeat(0, 2*l), 0, np.repeat(0, n)))
    h = matrix(h)
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    solvers.options['feastol'] = 1e-12
    solvers.options['reltol'] = 1e-12
    solvers.options['abstol'] = 1e-12

    sol = solvers.qp(Q, L, G, h, A, b)
    solution = sol['x'][:n]

    return np.asarray(solution).flatten()


def test_ssvr():
    X, Y, ground_truth = make_regression(
        shuffle=False, n_samples=500, n_features=25, n_informative=25,
        random_state=1541, n_targets=1, coef=True)
    ground_truth[ground_truth < 0] = 0.0
    ground_truth = ground_truth / np.sum(ground_truth)
    Y = X @ ground_truth
    C = 0.05
    nu = 0.77
    max_iter = 10000
    tol = 1e-16
    model = SSVR(nu=nu, C=C, tol=tol, max_iter=max_iter)
    model.fit(X, Y)
    coef = model.coef_
    coef_cvx = primal_nu_SVR_constrained(X, Y, cost=C, nu=nu)

    assert np.allclose(coef, coef_cvx)


if __name__ == '__main__':
    test_ssvr()
