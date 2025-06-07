from .env import CevicheEnv
import torch
import numpy as np

# env = CevicheEnv()
# env.print_state()
# t = np.ones((1,))
# t = 0.1 * t
# s = np.zeros((1,), dtype=np.int32)

# env.step([t, s])
# env.print_state()

def test():
    with open("log.txt", "w") as f:
        print("hello", file=f)

if __name__ == "__main__":
    test()