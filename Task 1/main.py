import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# === configs ===
TRAINING_PART = .9  # part of the data used for training
TEST_PART = .1  # part of the data used for testing


# === read training files ===
x_train_csv = pd.read_csv("X_train.csv")
x_train_vals = np.array(x_train_csv.values)
num_train_samples = np.shape(x_train_vals)[0]
x_train_data = x_train_vals[:, 1:]

y_train_csv = pd.read_csv("y_train.csv")
y_train_vals = np.array(y_train_csv.values)
y_train_data = y_train_vals[:, 1:]
y_train_data = y_train_data.flatten()


# === pre-process training data ===

# --- imputation ---
# compute means of rows, ignoring nan values
x_train_data_means = np.nanmean(x_train_data, axis=1)
# fill each empty cell with the mean of its row
x_train_data = np.array([np.array([xv if not np.isnan(xv) else x_train_data_means[i] for xv in xtd])
                         for i, xtd in enumerate(x_train_data)])

# --- outlier detection ---
# TODO


# --- feature selection ---
# TODO


# === divide data ===
training_indices = [i for i in range(num_train_samples) if np.random.uniform() <= TRAINING_PART]
x_train = x_train_data[training_indices]
y_train = y_train_data[training_indices]

test_indices = [i for i in range(num_train_samples) if np.random.uniform() <= TEST_PART]
x_test = x_train_data[test_indices]
y_test = y_train_data[test_indices]


# === fit model ===
model = LinearRegression()
model = model.fit(x_train, y_train)


# === output test loss ===
y_val = model.predict(x_test)
print(r2_score(y_test, y_val))


# === read prediction files ===
x_pred_csv = pd.read_csv("X_test.csv")
x_pred_vals = np.array(x_pred_csv.values)
pred_ids = x_pred_vals[:, 0]
x_pred_data = x_pred_vals[:, 1:]


# === pre-process prediction data ===
# compute means of rows, ignoring nan values
x_pred_data_means = np.nanmean(x_pred_data, axis=1)
# fill each empty cell with the mean of its row
x_pred_data = np.array([np.array([xv if not np.isnan(xv) else x_pred_data_means[i] for xv in xtd])
                         for i, xtd in enumerate(x_pred_data)])


# === make predictions ===
y_pred = model.predict(x_pred_data)


# === output predictions ===
out_df = pd.DataFrame({
    'Id': pd.Series(pred_ids, dtype="int32"),
    'y': pd.Series(y_pred.flatten(), dtype='float64')
})
out_df.to_csv("test_out.csv", index=False)
