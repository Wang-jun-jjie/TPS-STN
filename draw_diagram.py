import matplotlib.pyplot as plt
import numpy as np

ex2=[37.8252, 52.6431, 56.6952, 59.0548, 60.5957, 61.6729, 62.8418, 63.9191, 64.8679, 64.9747, \
     65.7534, 65.9302, 66.3637, 66.5538, 67.0091, 67.7578, 67.3059, 67.8045, 68.1580, 68.3214]
ex2_t=84.4852

ex5=[52.9899, 66.3004, 69.5421, 71.5098, 72.4270, 73.2524, 74.5064, 74.7715, 75.4452, 75.7521, \
     76.2240, 76.4841, 76.7393, 76.8510, 77.0911, 77.1795, 77.3679, 77.6881, 77.8265, 77.7748]
ex5_t=88.2011

ex8=[55.4262, 76.6509, 80.4029, 82.7108, 84.2349, 84.7219, 85.3439, 85.9375, 86.0259, 86.5678, \
     86.3044, 86.6929, 86.9697, 87.2148, 87.4133, 87.5467, 87.7368, 87.9069, 87.8952, 87.5984]
ex8_t=91.5565

x=range(1,21)
plt.xticks(range(0,22,2))
plt.plot(x, ex2, label='vanilla training', color='tab:blue')
plt.plot(x, ex5, label='deskewing training', color='tab:orange')
plt.plot(x, ex8, label='STN training', color='tab:green')
plt.plot(20, ex2_t, '<', label='vanilla testing', color='tab:blue');
plt.plot(19, ex5_t, 's', label='deskewing testing', color='tab:orange');
plt.plot(18, ex8_t, 'd', label='STN testing', color='tab:green');

plt.title('Classification performance on distorted MNIST')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend(loc = 'lower right')

plt.xlim(0, 21)
plt.ylim(35, 100)



plt.show()
plt.savefig('plot.png')
plt.close()