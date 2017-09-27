from __future__ import print_function

import sys
import torchfile
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
print ('Path: ' + path)
noiseStats = torchfile.load(path)

# Print noise stats
for i in range(len(noiseStats)):
	print ('Class {} ->'.format(i))
	for j in range(len(noiseStats[i])):
		print ('{}: {},'.format(j, noiseStats[i][j]), end='')
	print ('\n')

# Show bar chart
# class_ind = int(sys.argv[2])
num_classes = 100
# data = noiseStats[class_ind]
data = np.sum(noiseStats, axis=0)
ind = np.arange(num_classes)
width = 0.1

fig, ax = plt.subplots()
rects = ax.bar(ind, data, width, color='r')

ax.set_ylabel('Number of Corrupted Labels')
ax.set_xlabel('Class Corrupted To')
ax.set_xticks(ind + width/2)
ax.set_xticklabels([])

plt.show()