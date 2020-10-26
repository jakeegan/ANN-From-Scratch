import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    np.random.seed(1)

    # Load data
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

    # Train ANN
    ann = ANN(input_num=30,
              hidden_num=30,
              epoch=10000,
              batches=6,
              learning_rate=0.0001,
              momentum=0.7,
              dropout=0)
    ann.train(X_train, np.array([y_train]).T)

    # Test ANN
    y_pred = ann.predict(X_test)

    # Report results
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=["Malignant", "Benign"], columns=["Malignant", "Benign"])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
