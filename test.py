import os
import numpy as np

# import joblib
# loaded_rf = joblib.load("./random_forest.joblib")
# joblib.dump(loaded_rf, "./random_forest_compress.joblib", compress=6)

print(f"Uncompressed Random Forest: {np.round(os.path.getsize('random_forest.joblib') / 1024 / 1024, 2) } MB")