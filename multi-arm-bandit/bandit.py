import numpy as np
import matplotlib.pyplot as plt
BANDITS = 10
STEPS = 1000
TEST_RUNS = 1000
rng = np.random.default_rng()

def main():
    fig, ax = plt.subplots(2, 1)

    overall_rew_1 = np.zeros(shape=[STEPS])
    overall_rew_2 = np.zeros(shape=[STEPS])
    overall_rew_3 = np.zeros(shape=[STEPS])
    
    overall_per_1 = np.zeros(shape=[STEPS])
    overall_per_2 = np.zeros(shape=[STEPS])
    overall_per_3 = np.zeros(shape=[STEPS])

    for t in range(1, TEST_RUNS + 1):
        rewards_1, rewards_2, rewards_3, percent_1, percent_2, percent_3 = test_run()

        overall_rew_1 += 1 / t * (rewards_1 - overall_rew_1)
        overall_rew_2 += 1 / t * (rewards_2 - overall_rew_2)
        overall_rew_3 += 1 / t * (rewards_3 - overall_rew_3)

        overall_per_1 += 1 / t * (percent_1 - overall_per_1)
        overall_per_2 += 1 / t * (percent_2 - overall_per_2)
        overall_per_3 += 1 / t * (percent_3 - overall_per_3)

    ax[0].plot(np.arange(0, STEPS), overall_rew_1, label="0.1")
    ax[0].plot(np.arange(0, STEPS), overall_rew_2, label="0.01")
    ax[0].plot(np.arange(0, STEPS), overall_rew_3, label="0")

    ax[0].legend()

    ax[1].plot(np.arange(0, STEPS), overall_per_1, label="0.1")
    ax[1].plot(np.arange(0, STEPS), overall_per_2, label="0.01")
    ax[1].plot(np.arange(0, STEPS), overall_per_3, label="0")

    ax[1].legend()

    plt.tight_layout()
    plt.show()


def test_run():
    bandit_means = rng.normal(loc=0, scale=1, size=BANDITS)
    bandit_stds = np.ones(BANDITS)

    best_arm = np.argmax(bandit_means)

    rewards_1, percent_1 = eps_greedy(bandit_means, bandit_stds, best_arm, eps=0.1)
    rewards_2, percent_2 = eps_greedy(bandit_means, bandit_stds, best_arm, eps=0.01)
    rewards_3, percent_3 = eps_greedy(bandit_means, bandit_stds, best_arm, eps=0)

    return rewards_1, rewards_2, rewards_3, percent_1, percent_2, percent_3


def eps_greedy(bandit_means, bandit_stds, best_arm, eps=0.1):
    pred_vals = np.zeros(BANDITS)
    pred_counts = np.zeros(BANDITS)
    avg_reward = 0
    best_arm_cnt = 0
    reward_list = np.zeros(STEPS)
    percent_list = np.zeros(STEPS)

    for s in range(1, STEPS + 1):
        r = rng.random()

        if (r > eps):
            chosen = np.argmax(pred_vals)
        else:
            chosen = rng.integers(0, BANDITS)
        
        if (chosen == best_arm):
            best_arm_cnt += 1

        reward = rng.normal(loc=bandit_means[chosen], scale=bandit_stds[chosen])
        pred_counts[chosen] += 1

        avg_reward += 1 / s * (reward - avg_reward)
        pred_vals[chosen] += 1 / pred_counts[chosen] * (reward - pred_vals[chosen])

        reward_list[s - 1] = avg_reward
        percent_list[s - 1] = best_arm_cnt / s

    return reward_list, percent_list



if (__name__ == "__main__"):
    main()