import matplotlib.pyplot as plt

policies = ["LRU", "ML", "Hybrid"]

hit_rates = [0.82, 0.91, 0.92]

plt.figure()

plt.bar(policies, hit_rates)

plt.ylabel("Cache Hit Rate")
plt.title("Cache Policy Comparison")

plt.ylim(0,1)

plt.savefig("results/plots/cache_comparison.png")

plt.show()