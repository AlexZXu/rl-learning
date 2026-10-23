def main():
    game_state = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    V = {}

    action = 1
    while (True):
        value_sum = 0
        cnt = 0

        for i in range(0, 3):
            for j in range(0, 3):
                if (game_state[i][j] != 0):
                    continue

                game_state[i][j] = action

                if game_state not in V:
                    V[game_state] = 0

                value_sum += V[game_state]
                cnt += 1

                game_state[i][j] = 0

        final_val = value_sum / cnt

        break



def print_game(game_state):
    for row in game_state:
        for item in row:
            if (item == 0):
                print("_", end=" ")
            if (item == 1):
                print("X", end=" ")
            if (item == 2):
                print("O", end=" ")
        print()

def game_condition(game_state):
    for row in game_state:
        if (row[0] != 0 and row[0] == row[1] and row[1] == row[2]):
            return row[0]

    for i in range(0, 3):
        if (game_state[0][i] != 0 and game_state[0][i] == game_state[1][i] and game_state[1][i] == game_state[2][i]):
            return game_state[0][i]

    if (game_state[0][0] != 0 and game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2]):
        return game_state[0][0]

    if (game_state[0][2] != 0 and game_state[0][2] == game_state[1][1] and game_state[1][1] == game_state[2][0]):
        return game_state[0][2]

    return 0

if (__name__ == "__main__"):
    main()