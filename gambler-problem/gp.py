import numpy as np
import matplotlib.pyplot as plt

EPS = 0.000001
P_H = 0.4
def main():
    V_s = np.zeros(101)
    V_s[100] = 1

    fig, ax = plt.subplots(nrows=2, figsize=(10, 8))

    ax[0].plot(V_s)

    V_s = val_iter(V_s, ax[0])

    policy = determine_policy(V_s)

    ax[1].plot(policy)

    plt.tight_layout()
    plt.show()

def val_iter(V_s, ax):
    max_delta = EPS + 1

    while max_delta > EPS:
        max_delta = 0
        for s in range(1, 100):
            v = V_s[s]
            
            max_val = 0
            for a in range(1, min(s, 100 - s) + 1):
                new_val = P_H * V_s[s + a] + (1 - P_H) * V_s[s - a]

                max_val = max(max_val, new_val)

            V_s[s] = max_val
            max_delta = max(max_delta, abs(v - V_s[s]))
        ax.plot(V_s)

    return V_s        

def determine_policy(V_s):
    print(V_s)
    policy = np.zeros(101)
    for s in range(1, 100):
        best_action = 0
        max_val = 0

        for a in range(1, min(s, 100 - s) + 1):
            val = P_H * V_s[s + a] + (1 - P_H) * V_s[s - a]

            if (val > max_val + 1e-6):
                max_val = val
                best_action = a

        policy[s] = best_action

    return policy

if (__name__ == "__main__"):
    main()