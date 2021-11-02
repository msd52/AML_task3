import pandas as pd
import numpy as np

test_csv = pd.read_csv("test.csv")
test_data_comp = np.array(test_csv.values)
num_samples = np.shape(test_data_comp)[0]
test_ids = test_data_comp[:, 0]
test_data = test_data_comp[:, 1:]
test_y = np.mean(test_data, axis=1)

out_df = pd.DataFrame({
    'Id': pd.Series(test_ids, dtype="int32"),
    'y': pd.Series(test_y, dtype='float64')
})
out_df.to_csv("test_out.csv", index=False)
