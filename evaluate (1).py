import os
import json
import subprocess
import sys
import pathlib
import tarfile
import pandas as pd
import numpy as np

#from sagemaker.tensorflow import TensorFlow as tf

# def install_requirements():
#     try:
#         subprocess.check_call([
#             sys.executable, "-m", "pip", "install", "-r",
#             "/opt/ml/processing/requirement/requirements.txt"
#         ])
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install packages from requirements.txt: {e}")
#         sys.exit(1)

# # Install the dependencies
# install_requirements()


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])




if __name__ == "__main__":
    
    # model_path = f"opt/ml/processing/model/model.tar.gz"
    # with tarfile.open(model_path) as tar:
    #     tar.extractall(path=".")
    # model = tf.keras.models.load_model("./")
    install("tensorflow==2.11.0")
    import tensorflow as tf
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")

    model = tf.keras.models.load_model("./model/1")
    test_path = "/opt/ml/processing/test/"
    
    x_test = np.load(os.path.join(test_path, "x_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))
    # test = pd.read_csv(os.path.join(test_path, 'test.csv'))
    # x_test = test.iloc[:,1:]
    # y_test = test.iloc[:,0]    
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest MSE :", scores)

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "regression_metrics": {
            "mse": {"value": scores},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))