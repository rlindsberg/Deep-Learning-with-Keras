from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
# random_uniform: Weights are initialized to uniformly random small values in (-0.05, 0.05). In other words, any value within the given interval is equally likely to be drawn.
#   random_normal: Weights are initialized according to a Gaussian, with a zero mean and small standard deviation of 0.05. For those of you who are not familiar with a Gaussian, think about a symmetric bell curve shape.
#   zero: All weights are initialized to zero.

model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
