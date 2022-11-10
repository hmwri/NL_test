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
data3 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_2layer_h100_info.pkl")
print(data1)
data4 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_1layer_h100_info.pkl")
print(data1)


history_train1 = data1["history_train"]
history_eval1 = data1["history_eval"]
history_train2 = data2["history_train"]
history_eval2 = data2["history_eval"]
history_train3 = data3["history_train"]
history_eval3 = data3["history_eval"]
history_train4 = data4["history_train"]
history_eval4 = data4["history_eval"]

plot_history(history_train1,history_train2,history_train3,history_train4)
plot_history(history_eval1,history_eval2,history_eval3,history_eval4)


data5 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_1layer_notadvanced_info.pkl")
data6 = load_picke("/Users/taiyuu/Downloads/lstm_trainer_2layer_advanced_info.pkl")
history_train5 = data5["history_train"]
history_eval5 = data5["history_eval"]
history_train6 = data6["history_train"]
history_eval6 = data6["history_eval"]
plot_history(history_eval5,history_eval6)
plot_history(history_train5,history_train6)