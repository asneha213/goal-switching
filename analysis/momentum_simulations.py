import numpy as np
from matplotlib import pyplot as plt

class MomentumSim():

    """
    Simulations for momentum computations (Figure 6 in the manuscript)
    """

    def __init__(self, gamma, nu):
        self.alpha = 0.4
        self.gamma = gamma
        self.nu = nu
        self.prev_progress = 0
        self.v = 0.5
        self.b = 0.1

    def reset_model(self, gamma, nu):
        self.gamma = gamma
        self.nu = nu
        self.prev_progress = 0
        self.v = 0.5
        self.b = 0.1

    def update_params_prospective(self, flip):
        if flip == "e":
            self.v = self.v + self.alpha * (0 - self.v)
        else:
            self.v = self.v + self.alpha * (1 - self.v)
            self.prev_progress += self.nu


    def update_params_momentum(self, flip):
        if flip == 'e':
            progress_new = self.prev_progress
            progress = self.prev_progress
        else:
            progress_new = self.prev_progress + self.nu
            progress = self.prev_progress

        delta = self.gamma * (self.v* progress_new + self.b ) - self.v * (progress) - self.b

        self.v += self.alpha * delta * ( progress_new )
        self.b += self.alpha * delta

        self.prev_progress = progress_new

    def calculate_goal_value_momentum(self):
        val = self.v * self.prev_progress + self.b
        return val

    def calculate_goal_value_recursion(self, progress, count):
        """
        Prospective value computation
        """
        if count > 20:
            return 0
        if progress >= 1:
            return 2

        q = self.v * self.gamma * self.calculate_goal_value_recursion(progress + self.nu, count + 1) / (1 - (1 - self.v) * self.gamma)

        return q


    def calculate_goal_value_prospective(self):
        q = self.calculate_goal_value_recursion(self.prev_progress, 0)
        return q

    def simulate_prospective(self, flips, num_trials=100):
        progress = []
        qvals = []
        for i in range(num_trials):
            flip = flips[i]
            self.update_params_prospective(flip)
            qvals.append(self.calculate_goal_value_prospective())
            progress.append(self.prev_progress)
            if self.prev_progress >= 1:
                break

        return qvals, progress


    def simulate_momentum(self, flips, num_trials=100):
        progress = []
        qvals = []
        for i in range(num_trials):
            flip = flips[i]
            self.update_params_momentum(flip)
            if self.prev_progress >= 1:
                break
            qvals.append(self.calculate_goal_value_momentum())
            progress.append(self.prev_progress)
        return qvals, progress

    def simulate_momentum_qval_variation_with_gammas(self, num_trials=100):
        """
        Simulation how TD-momentum computed goal value varies with fractional progress for different gamma values
        Figure (6 A) in the manuscript
        """
        gammas =[0.6, 0.75, 0.9, 0.99]
        fig = plt.figure()
        p = 0.6
        flips = np.random.choice(['e', 'f'], p=[1 - p, p], size=num_trials)

        for gamma in gammas:
            self.reset_model(gamma, self.nu)
            qvals, progress = self.simulate_momentum(flips=flips, num_trials=num_trials)

            plt.plot(progress, qvals, label=r'$\gamma$ = ' + str(gamma), marker='o', linestyle='dashed')

        plt.xlabel('Progress', fontsize=16)
        plt.ylabel('Q-value (momentum)', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=15)

        plt.title(r'unit progress, $\eta = 0.14$', fontsize=15)
        #plt.savefig('figures/momentum_simulations.png')
        plt.show()


    def simulate_momentum_prospective_distinctions(self):
        """
        Simulate the distinction between momentum and prospective computations when there is a probability shift from 0.6 to 0.2
        Figure (6 B) in the manuscript
        """
        gamma = self.gamma
        self.reset_model(gamma, self.nu)
        flips_60 = np.random.choice(['e', 'f'], p=[0.4, 0.6], size=15)
        flips_20 = np.random.choice(['e', 'f'], p=[0.8, 0.2], size=25)
        qvals, progress = self.simulate_momentum(flips=flips_60, num_trials=15)
        self.v = 0.5
        self.b = 0.1
        qvals_2, progress_2 = self.simulate_momentum(flips=flips_20, num_trials=25)

        plt.plot(np.arange(len(qvals) + len(qvals_2)), qvals + qvals_2, label="momentum", marker='o', linestyle='dashed')


        self.reset_model(gamma, self.nu)
        qvals, progress = self.simulate_prospective(flips=flips_60, num_trials=15)
        self.v = 0.5
        self.b = 0.1
        qvals_2, progress_2 = self.simulate_prospective(flips=flips_20, num_trials=25)

        plt.plot(np.arange(len(qvals) + len(qvals_2)), qvals + qvals_2, label="prospective",\
                 marker='o', linestyle='dashed')

        plt.xlabel('Number of trials', fontsize=16)
        plt.ylabel('Q-value', fontsize=16)\

        # plot a step function to indicate the shift in probability from 0.6 to 0.2
        rounds = np.arange(50)
        probability = np.where(rounds < 16, 0.6, 0.2)

        # Plotting the step function
        plt.step(rounds, probability, where='post', color='black')

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=15)

        # add title indicating nu value (give nu in greek notation)
        plt.title(r'unit progress, $\eta = 0.05$', fontsize=15)
        #plt.savefig('figures/momentum_simulations_comparison_to_pros.png')
        plt.show()



if __name__ == '__main__':

    sim = MomentumSim(0.9, 0.14)
    #sim.simulate_momentum_prospective_distinctions()
    sim.simulate_momentum_qval_variation_with_gammas()

