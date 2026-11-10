import numpy as np
from math import factorial

MAX_CARS = 20
MAX_MOVE = 5
GAMMA = 0.9
EPS = 1e-4
RENT_REWARD = 10
MOVE_COST = 2

RENT_LAMBDA = (3, 4)
RETURN_LAMBDA = (3, 2)

N = MAX_CARS + 1  # states are 0..20 cars, so 21 of them
ACTIONS = np.arange(-MAX_MOVE, MAX_MOVE + 1)  # + means move from loc 1 to loc 2


def poisson_trunc(lam, k):
    # P(X = 0..k-1) for X ~ Poisson(lam), with the whole tail P(X >= k-1) lumped into the last bucket
    probs = np.array([lam ** n / factorial(n) * np.exp(-lam) for n in range(k)])
    probs[-1] += 1 - probs.sum()
    return probs


def location_model(rent_lambda, return_lambda):
    # P[n, n'] = prob a location that starts the day with n cars ends it with n'
    # R[n]     = expected rental income for a location that starts the day with n cars
    # The two locations never interact once the overnight move is done, so each gets
    # its own 21x21 model and the joint transition is just their outer product.
    P = np.zeros([N, N])
    R = np.zeros(N)

    reqs = np.arange(N)
    rent_probs = poisson_trunc(rent_lambda, N)  # asking for >= 20 cars is the same as asking for 20

    for n in range(N):
        rented = np.minimum(reqs, n)
        R[n] = RENT_REWARD * np.sum(rent_probs * rented)

        # distribution over cars left after rentals
        after_rent = np.bincount(n - rented, weights=rent_probs, minlength=N)

        # returns land on top of whatever is left, anything over 20 vanishes
        for left in range(n + 1):
            ret_probs = poisson_trunc(return_lambda, N - left)
            P[n, left:] += after_rent[left] * ret_probs

    return P, R


def main():
    P1, R1 = location_model(RENT_LAMBDA[0], RETURN_LAMBDA[0])
    P2, R2 = location_model(RENT_LAMBDA[1], RETURN_LAMBDA[1])

    V = np.zeros([N, N])
    pi = np.zeros([N, N], dtype=int)

    s1, s2 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    it = 0
    while True:
        V = pol_eval(V, pi, s1, s2, P1, R1, P2, R2)
        pi, stable = pol_improve(V, pi, s1, s2, P1, R1, P2, R2)
        print(f"policy {it}:")
        print(np.flipud(pi.T))  # rows = cars at loc 2 (20 at top), cols = cars at loc 1, like Figure 4.2
        it += 1
        if (stable):
            break

    print(f"V range: {V.min():.1f} .. {V.max():.1f}")


def q_values(V, s1, s2, a, P1, R1, P2, R2):
    # Value of taking action a in every state at once (shape (N, N)).
    # E[V(s')] for post-move state (m1, m2) is sum_{i,j} P1[m1,i] P2[m2,j] V[i,j],
    # which is exactly (P1 @ V @ P2.T)[m1, m2]. Two matmuls replace the four inner loops.
    m1 = np.clip(s1 - a, 0, MAX_CARS)
    m2 = np.clip(s2 + a, 0, MAX_CARS)
    EV = P1 @ V @ P2.T
    return -MOVE_COST * np.abs(a) + R1[m1] + R2[m2] + GAMMA * EV[m1, m2]


def pol_eval(V, pi, s1, s2, P1, R1, P2, R2):
    m1 = np.clip(s1 - pi, 0, MAX_CARS)
    m2 = np.clip(s2 + pi, 0, MAX_CARS)
    reward = -MOVE_COST * np.abs(pi) + R1[m1] + R2[m2]

    while True:
        EV = P1 @ V @ P2.T
        new_V = reward + GAMMA * EV[m1, m2]
        max_change = np.max(np.abs(new_V - V))
        V = new_V
        if (max_change < EPS):
            return V


def pol_improve(V, pi, s1, s2, P1, R1, P2, R2):
    Q = np.stack([q_values(V, s1, s2, a, P1, R1, P2, R2) for a in ACTIONS])  # Q shape: (actions, N, N)

    # can't move more cars than you have
    valid = ((s1 - ACTIONS[:, None, None]) >= 0) & ((s2 + ACTIONS[:, None, None]) >= 0)
    Q[~valid] = -np.inf

    new_pi = ACTIONS[np.argmax(Q, axis=0)]
    # only switch if strictly better, otherwise ties between equal actions make it flip forever
    keep = Q[pi + MAX_MOVE, s1, s2] >= Q.max(axis=0) - 1e-9
    new_pi = np.where(keep, pi, new_pi)
    return new_pi, np.array_equal(new_pi, pi)


if (__name__ == "__main__"):
    main()
