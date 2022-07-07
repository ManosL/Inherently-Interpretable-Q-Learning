import os
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from statistics import mean
import csv
from datetime import datetime

if not os.path.exists('results'):
    os.mkdir('results')

ACTIONS_CSV_PATH = "/actions.csv"
SCORES_CSV_PATH = "/scores.csv"
OBSERVATIONS_CSV_PATH = "/observations.csv"
SCORES_PNG_PATH = "./scores.png"
ACCUMULATIVE_SCORES_CSV_PATH = "/accumulative_scores.csv"
ACCUMULATIVE_SCORES_PNG_PATH = "/accumulative_scores.png"
SOLVED_CSV_PATH = "/solved.csv"
SOLVED_PNG_PATH = "/solved.png"
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 10


class ScoreLogger:

    def __init__(self, params):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.params = params
        self.acc_x = []
        self.acc_y = []
        self.acc_stdev = []

        timestamp = datetime.now()

        self.folder_path = f'results/experiment_{self.params["category"]}_{timestamp.year}{timestamp.strftime("%m")}{timestamp.strftime("%d")}/'

        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        
        if os.path.exists(self.folder_path+ACTIONS_CSV_PATH):
            os.remove(self.folder_path+ACTIONS_CSV_PATH)
        if os.path.exists(self.folder_path+OBSERVATIONS_CSV_PATH):
            os.remove(self.folder_path+OBSERVATIONS_CSV_PATH)

        if os.path.exists(self.folder_path+SCORES_PNG_PATH):
            os.remove(self.folder_path+SCORES_PNG_PATH)
        if os.path.exists(self.folder_path+SCORES_CSV_PATH):
            os.remove(self.folder_path+SCORES_CSV_PATH)

        if os.path.exists(self.folder_path+ACCUMULATIVE_SCORES_CSV_PATH):
            os.remove(self.folder_path+ACCUMULATIVE_SCORES_CSV_PATH)
        if os.path.exists(self.folder_path+ACCUMULATIVE_SCORES_PNG_PATH):
            os.remove(self.folder_path+ACCUMULATIVE_SCORES_PNG_PATH)

    def add_score(self, score, run):
        self._save_csv(self.folder_path+SCORES_CSV_PATH, score)
        self._save_csv(self.folder_path+ACCUMULATIVE_SCORES_CSV_PATH, score)
        self._save_png(input_path=self.folder_path+SCORES_CSV_PATH,
                       output_path=self.folder_path+SCORES_PNG_PATH,
                       x_label="episodes",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=False,
                       show_trend=True,
                       show_legend=True)
        self._save_cummulative_score(self.folder_path+ACCUMULATIVE_SCORES_CSV_PATH,
                       output_path=self.folder_path+ACCUMULATIVE_SCORES_PNG_PATH,
                       x_label="episodes",
                       y_label="scores",
                       average_of_n_last=10,
                       show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run-CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(self.folder_path+SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=self.folder_path+SOLVED_CSV_PATH,
                           output_path=self.folder_path+SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)

    
    def _save_cummulative_score(self, input_path, output_path, x_label, y_label, average_of_n_last, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            j = 0
            for i in range(0, len(data)):
                if len(data[i]) == 0:
                    continue
                x.append(int(j))
                y.append(int(data[i][0]))
                j+=1

        if (x[-1] > 1) and (x[-1] % average_of_n_last == 0):
            plt.subplots()
            average_range = average_of_n_last if average_of_n_last is not None else len(x)
            self.acc_x.append(x[-average_range:][0])
            self.acc_y.append(np.mean(y[-average_range:]))
            self.acc_stdev.append(np.std(y[-average_range:]) / np.sqrt(np.size(y[-average_range:])))
            plt.plot(self.acc_x, self.acc_y, label=str(average_range) + " runs average")
            
            plt.fill_between(self.acc_x, np.array(self.acc_y) - np.array(self.acc_stdev), np.array(self.acc_y) + np.array(self.acc_stdev), alpha=0.5)


            parameters = f"""
                        - Learning rate: {self.params['learning_rate']}
                        - ε: {self.params['eps']}, {self.params['eps_decay']} decay
                        - γ: {self.params['gamma']}
                        - Batch size: {self.params['batch_size']}
                        - Bins: {self.params['bins']}
                        - Epochs: {self.params['epochs']}
                        - With Prioritized Experience Replay: {self.params['prioritized_experience_replay']}
                        """

            if self.params['target_model_updates']:
                parameters += f"    - Target SGTs model update every {self.params['target_model_updates']} episodes"

            plt.text(-0.3, 0.8, parameters, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
            plt.title(f"{self.params['method']}, Environment: {self.params['env_name']}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            if show_legend:
                plt.legend(loc="upper left")
            
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

    def add_action(self, action):
        if not os.path.exists(self.folder_path+ACTIONS_CSV_PATH):
            with open(self.folder_path+ACTIONS_CSV_PATH, "w"):
                pass
        actions_file = open(self.folder_path+ACTIONS_CSV_PATH, "a")
        with actions_file:
            writer = csv.writer(actions_file)
            if type(action == int):
                writer.writerow(np.array([action]).reshape(1, -1)[0])
            else:
                writer.writerow(np.array(action).reshape(1, -1)[0])

    def add_observation(self, observation):
        if not os.path.exists(self.folder_path+OBSERVATIONS_CSV_PATH):
            with open(self.folder_path+OBSERVATIONS_CSV_PATH, "w"):
                pass
        obs_file = open(self.folder_path+OBSERVATIONS_CSV_PATH, "a")
        with obs_file:
            writer = csv.writer(obs_file)
            writer.writerow(observation)
    
    
    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            j = 0
            for i in range(0, len(data)):
                if len(data[i]) == 0:
                    continue
                x.append(int(j))
                y.append(int(data[i][0]))
                j+=1

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        parameters = f"""
                        - Learning rate: {self.params['learning_rate']}
                        - ε: {self.params['eps']}, {self.params['eps_decay']} decay
                        - γ: {self.params['gamma']}
                        - Batch size: {self.params['batch_size']}
                        - Bins: {self.params['bins']}
                        - Epochs: {self.params['epochs']}
                        - With Prioritized Experience Replay: {self.params['prioritized_experience_replay']}
                        """

        if self.params['target_model_updates']:
            parameters += f"    - Target SGTs model update every {self.params['target_model_updates']} episodes"

        plt.text(-0.3, 0.8, parameters, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        plt.title(f"{self.params['method']}, Environment: {self.params['env_name']}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])