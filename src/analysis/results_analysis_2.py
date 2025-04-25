import pickle

with open('src/results/results_2.pkl', 'rb') as f:
    data = pickle.load(f)
    print(type(data))  # This will print the data type