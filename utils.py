import pickle

def load_q_table(q_table_path="q_table.pkl"):
    try:
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        q_table = {}
    return q_table

def store_q_table(q_table, q_table_path="q_table.pkl"):
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
