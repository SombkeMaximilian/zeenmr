import numpy as np
import matplotlib.pyplot as plt
import metabodecon as md


def main():
    lorentzian = md.Lorentzian(
        sf=1.0,
        hw=1.0,
        maxp=0.0
    )

    print(lorentzian.sf)
    print(lorentzian.hw)
    print(lorentzian.maxp)
    lorentzian.sf = 2.0
    lorentzian.hw = 1.5
    print(lorentzian.sf)
    print(lorentzian.hw)
    print(lorentzian.maxp)

    x = np.linspace(-10, 10, 10000)
    y = lorentzian.evaluate_vec(x)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(x, y)
    plt.show()

    sf = [1.0, 2.0, 1.0]
    hw = [0.1, 0.15, 0.1]
    maxp = [4.5, 5.0, 5.5]
    lorentzians = [md.Lorentzian(sf=sf, hw=hw, maxp=maxp) for sf, hw, maxp in zip(sf, hw, maxp)]
    x = np.linspace(0, 10, 100000)
    y = md.Lorentzian.par_superposition_vec(x, lorentzians)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
