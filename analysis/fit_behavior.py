import optuna

from models import *


def get_optuna_params(trial, model_name):
    if model_name == "rescorla":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_c = trial.suggest_uniform('beta_c', 0.0, 10.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_g': beta_g, 'beta_c': beta_c}
        return params

    elif model_name == "td_persistence":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -2.0, 2.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a}


    elif model_name == "prospective" or model_name == "momentum":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -2.0, 2.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}


    elif model_name == "prospective_dl_momentum":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        alpha_e = trial.suggest_uniform('alpha_e', 0.0, 1.0)
        alpha_m = trial.suggest_uniform('alpha_m', 0.0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'alpha_e': alpha_e, 'alpha_m': alpha_m}

    elif model_name == "prospective_dl":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        alpha_e = trial.suggest_uniform('alpha_e', 0.0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'alpha_e': alpha_e}

    elif model_name == "prospective_momentum":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        alpha_e = trial.suggest_uniform('alpha_e', 0.0, 1.0)
        alpha_m = trial.suggest_uniform('alpha_m', 0.0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'alpha_e': alpha_e, 'alpha_m': alpha_m}

    elif model_name == "prospective_hyperbolic":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}

    elif model_name == "hybrid" or model_name == "hybrid_mp":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        wa = trial.suggest_uniform('wa', 0.0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'wa': wa}


    elif model_name == "prospective_just_goal" or model_name == "momentum_just_goal":
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
        beta_c = trial.suggest_uniform('beta_c', 0.0, 10)
        beta_0 = trial.suggest_uniform('beta_0', -1.0, 1.0)
        beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)

        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_c': beta_c, 'beta_0': beta_0, 'beta_g': beta_g, 'gamma': gamma}


    return params


class BehaviorFits:
    def __init__(self, experiment, subject_id, model_name, model_res=None, rewrite=False):
        self.experiment = experiment
        self.subject_id = subject_id
        self.model_name = model_name
        if not model_res:
            self.subject_data = get_subject_data_from_id(experiment, subject_id)
        else:
            self.subject_data = model_res

        self.test_blocks = None
        self.trt =None
        self.rewrite = rewrite

    def get_cv_blocks(self, num_blocks, num_folds, seed):
        np.random.seed(seed)
        blocks = np.arange(num_blocks)
        np.random.shuffle(blocks)
        blocks = np.array_split(blocks, num_folds)
        return blocks

    def get_choice_likelihood(self, params, test=False):

        blocks_data = self.subject_data['GoalSwitching']
        loglikelihoods = []
        action_loglikelihoods = []
        goal_loglikelihoods = []

        model = get_model(self.model_name, params)
        model.reset_slots()

        for block_num in range(len(blocks_data.keys())):

            if test == False and self.test_blocks is not None:
                if block_num in self.test_blocks:
                    continue
            if test:
                if block_num not in self.test_blocks:
                    continue

            if self.trt == 0:
                if block_num < len(blocks_data.keys()) / 2:
                    continue
            elif self.trt == 1:
                if block_num >= len(blocks_data.keys()) / 2:
                    continue

            block = blocks_data[str(block_num)]
            model.reset_card_probs()

            for trial_num in range(30):
                model.trial_num = trial_num
                trial = block['trial_' + str(trial_num)]
                action_prob, goal_prob = model.run_subject_action_trial(trial)
                
                action_loglikelihoods.append(np.log(action_prob))
                if goal_prob != -1:
                    goal_loglikelihoods.append(np.log(goal_prob))


        loglikelihood = - np.sum(action_loglikelihoods) - np.sum(goal_loglikelihoods)
        print(loglikelihood)
        return loglikelihood

    def get_optuna_objective(self, trial):
        params = get_optuna_params(trial, self.model_name)
        err = self.get_choice_likelihood(params)
        return err

    def fit_optuna(self, trt=None):
        best_params = []
        best_vals = []
        self.trt = trt

        for i in range(20):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=100)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals, best_params)
        return best_params, best_vals

    def fit_behavior(self, write=True):

        params, fits = self.fit_optuna()

        model_fits = {}
        model_fits['fits'] = fits
        model_fits['params'] = params

        if write:
            folder_name = 'results_latent/' + self.model_name + "_" + str(
                self.experiment) + "/"
            #if not os.path.exists(folder_name):
            #    os.makedirs(folder_name)

            file_name = folder_name + str(self.subject_id) + ".pkl"
            if os.path.exists(file_name):
                return None, None
            file = open(folder_name + str(subject_id) + ".pkl", "wb")
            pickle.dump(model_fits, file)
            file.close()

        return params, fits

    def get_cv_score(self, seed):
        if self.experiment == 1:
            num_folds = 3
            num_blocks = 18
        elif self.experiment == 2:
            num_folds = 3
            num_blocks = 12
        elif self.experiment == 4:
            num_folds = 3
            num_blocks = 12

        cv_blocks = self.get_cv_blocks(num_blocks, num_folds, seed)
        self.test_blocks = cv_blocks[0]
        print(self.test_blocks)
        params, fits = self.fit_optuna()
        cv_score = self.get_choice_likelihood(params, test=True)
        return cv_score

    def get_cv_results(self, seed):
        file_name = 'results_latent/' + self.model_name + "_" + str(self.experiment) + "/" + "cv_" + str(self.subject_id) + "_seed_" + str(seed) + ".pkl"
        if os.path.exists(file_name):
            return None
        cv_score = self.get_cv_score(seed)
        file = open(file_name, "wb")
        pickle.dump(cv_score, file)
        file.close()

    def get_test_retest_results(self, trt=0):
        params, fits = self.fit_optuna(trt=trt)
        params = list(params.values())
        model_fits = {}
        model_fits['fits'] = fits
        model_fits['params'] = params
        file = open('results/' + self.model_name + "_trt_" + str(self.experiment) + "/" + str(trt) + "/" + str(self.subject_id) + ".pkl", "wb")
        pickle.dump(model_fits, file)
        file.close()


if __name__ == "__main__":

    machine = "local"
    #machine = "server"


    cv = False
    #cv = True


    if machine == "local":
        experiment = 2
        subject_num = 4
        model_name = "prospective"
        #model_name = "rescorla"
        if cv:
            seed = 10
    else:
        experiment = int(sys.argv[1])
        subject_num = int(sys.argv[2])
        model_name = sys.argv[3]

        if cv:
            seed = int(sys.argv[4])

    if experiment == 3:
        experiment = "instr_1"


    subject_names = get_experiment_subjects(experiment)

    subject_id = subject_names[subject_num]

    print(subject_id)


    modelopt_t = BehaviorFits(experiment=experiment, subject_id=subject_id, \
                              model_name=model_name)

    if not cv:
        params, fits = modelopt_t.fit_behavior()
    else:
        modelopt_t.get_cv_results(seed)

