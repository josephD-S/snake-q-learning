import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores, max_q_values):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title('Scores')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.subplot(1, 2, 2)
    plt.title('Average Q-values')
    plt.xlabel('Number of Games')
    plt.ylabel('Average Q-value')
    plt.plot(max_q_values)
    plt.ylim(ymin=0)
    plt.draw()
    plt.pause(0.001)  # This is critical for rendering
