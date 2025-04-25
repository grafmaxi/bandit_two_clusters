import pickle

with open('src/results/results_3.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[0][0].shape)  # This will print the data type