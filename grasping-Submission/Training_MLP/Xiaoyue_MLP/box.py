import numpy as np
def random_box_positions():
    value1 = np.linspace(-0.6, 0.6, 120)
    value2 = [value for value in value1 if (0.35 <= value <= 0.6)]
    value1 = [value for value in value1 if (-0.3 <= value <= 0.3)]

    x = np.random.choice(value1)
    y = np.random.choice(value2)
    # x = -0.4
    # z = 0.5  # fixed box position, right in front of manipulator.

    return x, y
if __name__ == "__main__":
    for i in range(20):
        x, z = random_box_positions()
        print(f"{x},0.05,{z},0.0")