import matplotlib.pyplot as plt
import pickle

def get_q_losses_curve(q_losses_file_path):
    q_losses_file = open(q_losses_file_path, 'r')

    q1_curve = []
    q2_curve = []

    while 1:
        curr_line = q_losses_file.readline()

        if len(curr_line) == 0:
            break
        
        if curr_line.startswith('Q1 LOSS'):
            q1_curve.append(float(curr_line.strip().split()[2]))
        
        if curr_line.startswith('Q2 LOSS'):
            q2_curve.append(float(curr_line.strip().split()[2]))
    
    q_losses_file.close()
    assert len(q1_curve) == len(q2_curve)

    return q1_curve, q2_curve


def get_evaluation_curve(curve_file_path):
    curve_file = open(curve_file_path, 'rb')

    curve = pickle.load(curve_file)
    curve_file.close()

    print(curve)
    return curve


def plot_q_loss_curves(q1_curve, q2_curve, plot_title):
    smoothed_q1_curve = list(q1_curve)
    smoothed_q2_curve = list(q2_curve)

    for i in range(50, len(q1_curve)):
        smoothed_q1_curve[i] = sum(q1_curve[i-50:i+1]) / len(q1_curve[i-50:i+1])
        smoothed_q2_curve[i] = sum(q2_curve[i-50:i+1]) / len(q2_curve[i-50:i+1])

    print(len(q1_curve), len(smoothed_q1_curve))
    plt.plot(list(range(len(q1_curve))), smoothed_q1_curve, label='q1 loss')
    plt.plot(list(range(len(q2_curve))), smoothed_q2_curve, label='q2 loss')
    plt.title(plot_title)
    plt.legend()
    plt.show()

    return


def plot_evaluation_curve():
    pass


def main():
    nn_implementation_logs_path = '../NeuralNetworkTD3/logs/Hopper_nn_show_loss.txt'
    gbr_implementation_logs_path = './logs/final_exps/Hopper_gbr.txt'

    nn_q1_curve, nn_q2_curve   = get_q_losses_curve(nn_implementation_logs_path)
    gbr_q1_curve, gbr_q2_curve = get_q_losses_curve(gbr_implementation_logs_path)

    plot_q_loss_curves(nn_q1_curve,  nn_q2_curve,  'Neural Networks')
    plot_q_loss_curves(gbr_q1_curve, gbr_q2_curve, 'GBR')

    get_evaluation_curve('./eval_curves/Hopper_gbr/curve_1.pkl')
    return 0


if __name__ == '__main__':
    main()