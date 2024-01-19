import numpy as np


def get_random_walks_token_probs(num_trials=500, walk=True):

    token1 = np.random.uniform(0.2, 0.3)
    token2 = np.random.uniform(0.2, 0.3)
    token3 = np.random.uniform(0.7, 0.8)
    # Initialize initial probabilities for three tokens (between 0 and 1)
    token_probabilities = [token1, token2, token3]

    # Lists to store the probabilities over time
    token1_probabilities = [token_probabilities[0]]
    token2_probabilities = [token_probabilities[1]]
    token3_probabilities = [token_probabilities[2]]

    if walk==True:
        sd = 0.025
        # Perform random walks for 500 trials
        for _ in range(num_trials):
            # Generate random changes for each token's probability
            while(1):
                delta1 = np.random.normal(0, sd)
                delta2 = np.random.normal(0, sd)
                delta3 = np.random.normal(0, sd)

                if token_probabilities[0] + delta1 < 0 or token_probabilities[1] + delta2 < 0 or token_probabilities[2] + delta3 < 0:
                    continue
                elif token_probabilities[0] + delta1 > 1 or token_probabilities[1] + delta2 > 1 or token_probabilities[2] + delta3 > 1:
                    continue
                else:
                    break

            # Update probabilities
            token_probabilities[0] += delta1
            token_probabilities[1] += delta2
            token_probabilities[2] += delta3

            # Append the current probabilities to the respective lists
            token1_probabilities.append(token_probabilities[0])
            token2_probabilities.append(token_probabilities[1])
            token3_probabilities.append(token_probabilities[2])

    elif walk=="Slow":
        for trial_num in range(num_trials):
            if trial_num % 30 == 0:
                np.random.shuffle(token_probabilities)

            token1_probabilities.append(token_probabilities[0])
            token2_probabilities.append(token_probabilities[1])
            token3_probabilities.append(token_probabilities[2])

    elif walk=="Fast":
        for trial_num in range(num_trials):
            if trial_num % 5 == 0:
                np.random.shuffle(token_probabilities)

            token1_probabilities.append(token_probabilities[0])
            token2_probabilities.append(token_probabilities[1])
            token3_probabilities.append(token_probabilities[2])

    return token1_probabilities, token2_probabilities, token3_probabilities
