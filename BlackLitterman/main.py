import BlackLitterman
import numpy as np

def main():
    bl = BlackLitterman.BlackLitterman('BlackLittermanData.csv')
    P = np.array([[1,-1,0],[0,0,1]])
    Q = np.array([0.020, 0.015]).reshape(2,1)
    opt_weights, sharpe = bl.black_litterman_weights(3, 0.1, [0.5,0.4,0.1], P, Q, 0.1, 1)


main()