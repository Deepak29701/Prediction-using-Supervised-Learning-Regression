# Deepak Sarangi
# Prediction Using Supervised Learning Regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def estimate_coef(x,y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y*x)-(n*m_y*m_x)
    SS_xx = np.sum(x*x)-(n*m_x*m_x)

    b_1 = SS_xy/SS_xx
    b_0 = m_y-(b_1*m_x)
    return (b_0,b_1)
def plot_regression_line(x,y,b):
    plt.scatter(x,y)
    y_pred = b[0]+(b[1]*x)
    plt.plot(x,y_pred,color = "g")
    plt.xlabel('Hours')
    plt.ylabel('Score')
    plt.show()
def predict_score(p,b):
    pred = b[0]+(b[1]*p)
    print("Prediction of Score according to no. of study hours:",pred)
def main():
    data = pd.read_csv("Dataset.csv")
    x = data['Hours']
    y = data['Score']
    b = estimate_coef(x,y)
    print(b[0],b[1])
    plot_regression_line(x,y,b)
    p = float(input("Enter the number of study hours:"))
    predict_score(p,b)    
if __name__ == "__main__":
    main()

