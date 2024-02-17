import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_csv_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        df.index = pd.to_datetime(df['dateTime'], format='%Y-%m-%dT%H:%M:%SZ')
        temp = df['values']

        make_predictions(df)


def make_predictions(df):
    window_size = 1
    X, y = df_to_X_y(df['values'], window_size)
    train_predictions = model.predict(X).flatten()
    train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y})

    ax.clear()
    ax.plot(train_results['Train Predictions'], label='Train Predictions')
    ax.plot(train_results['Actuals'], label='Actuals')
    ax.set_title('Train Predictions vs Actuals')
    ax.legend()
    canvas.draw()


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


model = load_model('model2.h5')


root = tk.Tk()
root.title("Prediction")


fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


load_button = ttk.Button(root, text="Choose CSV", command=load_csv_data)
load_button.pack()

root.mainloop()
