import pickle

def load_qtable(qtable_path="qtable.pkl"):
    try:
        with open(qtable_path, "rb") as f:
            qtable = pickle.load(f)
    except FileNotFoundError:
        qtable = {}
    return qtable

def store_qtable(qtable, qtable_path="qtable.pkl"):
    with open(qtable_path, "wb") as f:
        pickle.dump(qtable, f)
