import pandas as pd
import numpy as np


x_train_csv = pd.read_csv("X_train.csv")
x_train_vals = np.array(x_train_csv.values)
num_train_samples = np.shape(x_train_vals)[0]
x_train_data = x_train_vals[:, 1:]

y_train_csv = pd.read_csv("y_train.csv")
y_train_vals = np.array(y_train_csv.values)
y_train_data = y_train_vals[:, 1:]

x_test_csv = pd.read_csv("X_test.csv")
x_test_vals = np.array(x_test_csv.values)
test_ids = x_test_vals[:, 0]
x_test_data = x_test_vals[:, 1:]

out_df = pd.DataFrame({
    'Id': pd.Series(test_ids, dtype="int32"),
    'y': pd.Series(y_train_data.flatten(), dtype='float64')
})
out_df.to_csv("test_out.csv", index=False)
