import numpy as np
import matplotlib.pyplot as plt



def plot_learning_and_eval_curve(x, train_scores, eval_scores, figure_file):
    train_running_avg = np.zeros(tuple([len(train_scores), len(train_scores[0])]))
    eval_running_avg  = np.zeros(tuple([len(eval_scores),  len(eval_scores[0])]))

    print(train_running_avg.shape)
    print(eval_running_avg.shape)
    assert(train_running_avg.shape == eval_running_avg.shape)

    for i in range(len(train_scores)):
        for j in range(len(train_scores[i])):
            train_running_avg[i][j] = np.mean(train_scores[i][max(0, j-50):(j+1)])
            eval_running_avg[i][j]  = eval_scores[i][j]

    train_means = np.zeros(len(train_scores[0]))
    eval_means  = np.zeros(len(eval_scores[0]))

    train_stds  = np.zeros(len(train_scores[0]))
    eval_stds   = np.zeros(len(eval_scores[0]))

    for i in range(len(train_running_avg[0])):
        train_means[i] = np.mean(train_running_avg[:, i])
        eval_means[i]  = np.mean(eval_running_avg[:, i])

        train_stds[i]  = np.std(train_running_avg[:, i])
        eval_stds[i]  = np.std(eval_running_avg[:, i])

    plt.plot(x, train_means, color='b', label='Training')
    plt.fill_between(x, (train_means - train_stds), (train_means + train_stds), color='b', alpha=.1)

    plt.plot(x, eval_means, color='r', label='Evaluation')
    plt.fill_between(x, (eval_means - eval_stds), (eval_means + eval_stds), color='r', alpha=.1)

    title =  'Running average of previous 50 scores and the \n'
    title += 'evaluation scores along with their standard deviation'
    plt.title(title)

    plt.legend()
    
    plt.savefig(figure_file)
    plt.clf()

    return



def plot_eval_curve(results, timestamps, results_figure_file, plot_title):
    res_means = []
    res_stds  = []

    for rs in results:
        assert(len(rs) == len(timestamps))

    for i in range(len(timestamps)):
        curr_results = []

        for j in range(len(results)):
            curr_results.append(sum(results[j][max(0, i-2):i+1]) / len(results[j][max(0, i-2):i+1]))
        
        res_means.append(np.mean(np.array(curr_results)))
        res_stds.append(0.5 * np.std(np.array(curr_results)))

    res_means = np.array(res_means)
    res_stds  = np.array(res_stds)
    
    plt.plot(timestamps, res_means, color='b')
    plt.fill_between(timestamps, (res_means - res_stds), (res_means + res_stds), color='b', alpha=.1)

    plt.title(plot_title)
    
    plt.savefig(results_figure_file)
    plt.clf()
    return



def plot_fit_curve(results, results_figure_file, plot_title):
    res_means = []
    res_stds  = []

    max_index  = max([len(ls) for ls in results])

    for i in range(max_index):
        curr_results  = []

        for j in range(len(results)):
            if i >= len(results[j]):
                continue
            
            curr_results.append(sum(results[j][max(0, i-50):i+1]) / len(results[j][max(0, i - 50):i+1]))
        
        res_means.append(float(np.mean(np.array(curr_results))))
        res_stds.append(float(np.std(curr_results)))

    res_means = np.array(res_means)
    res_stds  = np.array(res_stds)

    x = list(range(len(res_means)))

    plt.plot(x, res_means, color='b')
    plt.fill_between(x, (res_means - res_stds), (res_means + res_stds), color='b', alpha=.1)

    plt.title(plot_title)
    
    plt.savefig(results_figure_file)
    plt.clf()

    return
