def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson, norm
    plt.rcParams.update({
    'font.size': 18,
    })

    x = np.arange(0, 20)
    lambda_ = [0.5, 1, 2, 3]

    plt.figure(figsize=(12, 6))
    print(1-poisson.cdf(1, lambda_[0]),poisson.pmf([0,1], lambda_[0]))
    
    # I want the plot to range from 0 to 7
    for i in range(4):
        plt.plot(x, poisson.pmf(x, lambda_[i]), '-o', label='$\lambda$ = {}'.format(lambda_[i]))      
        
    plt.title('Poisson Distribution Example')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.xlim(0,7)

    plt.tight_layout()
    plt.savefig('DataPlotting/Poisson/PoissonExample2.pdf')
    return

main()