import pickle
import matplotlib.pyplot as plt

def load_picke(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def plot_history(*histories):
    for history in histories:
        plt.plot(list(range(1,len(history)+1)),history)
    plt.show()


data1 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_2_short_info.pkl")
print(data1)
data2 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_2layer_info.pkl")
print(data1)

history_train1 = data1["history_train"]
history_eval1 = data1["history_eval"]
history_train2 = data2["history_train"]
history_eval2 = data2["history_eval"]

plot_history(history_train1,history_train2)
plot_history(history_eval1,history_eval2)