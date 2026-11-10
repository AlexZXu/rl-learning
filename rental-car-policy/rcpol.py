import numpy as np
import math
EPS = 0.1

def main():
    V_s = np.zeros([21, 21])
    pi_s = np.zeros([21, 21], dtype=np.int16)

    pol_eval(V_s, pi_s)

def pol_eval(V_s, pi_s):
    rent_A_proba = [3**n /(math.factorial(n)) * math.e**(-3) for n in range(0, 21)]
    rent_B_proba = [4**n /(math.factorial(n)) * math.e**(-4) for n in range(0, 21)]
    return_A_proba = [3**n /(math.factorial(n)) * math.e**(-3) for n in range(0, 21)]
    return_B_proba = [2**n /(math.factorial(n)) * math.e**(-2) for n in range(0, 21)]

    max_change = EPS + 1
    while (max_change > EPS):
        max_change = 0

        for s1 in range(0, 21):
            for s2 in range(0, 21):
                v = V_s[s1, s2]
                action = pi_s[s1, s2]

                new_s1 = s1 - action
                new_s2 = s2 + action

                new_v = 0

                for turn_1 in range(0, 21 - new_s1):
                    for rent_1 in range(0, new_s1 + turn_1):
                        for turn_2 in range(0, 21 - new_s2):
                            for rent_2 in range(0, new_s2 + turn_2):
                                prob = (rent_A_proba[rent_1] * rent_B_proba[rent_2]
                                        * return_A_proba[turn_1] * return_B_proba[turn_2])

                                s1_prime = new_s1 + turn_1 - rent_1
                                s2_prime = new_s2 + turn_2 - rent_2

                                reward = -2 * np.abs(action) + 10 * (rent_1 + rent_2)

                                new_v += prob * (reward + V_s[s1_prime, s2_prime])

                V_s[s1, s2] = new_v

                max_change = max(max_change, np.abs(new_v - v))

    print(V_s)
    
if (__name__ == "__main__"):
    main()