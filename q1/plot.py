import matplotlib.pyplot as plt

"""
Total Proportion of variance explained by the first 1 principal components:0.2833447489537038
Total Proportion of variance explained by the first 10 principal components:0.6870788394291539
Total Proportion of variance explained by the first 50 principal components:0.8569321884808176
Total Proportion of variance explained by the first 100 principal components:0.908444723254203
Total Proportion of variance explained by the first 500 principal components:0.9806486239270209"""


plt.plot([1,10,50,100,500],[0.2833447489537038,0.6870788394291539,0.8569321884808176,0.908444723254203,0.9806486239270209])

plt.ylabel('PVE')
plt.xlabel('number of first principal components')

plt.suptitle('k vs PVE')
plt.savefig("kvspve.png")
plt.show()