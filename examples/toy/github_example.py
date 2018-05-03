# See issue: https://github.com/SheffieldML/GPy/issues/633
import numpy as np
import GPy

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y = np.array([[0, 0, 1, 0, 1, 0, 1, 1]]).T
print(X.shape)  # (8, 3)
# print(Y.shape) # (8, 1)

num_tasks = 2
num_obs, num_feats = X.shape
num_feats -= 1  # important, because the last column indicates the "task"

kern = GPy.kern.RBF(2) ** GPy.kern.Coregionalize(input_dim=1,
                                                 output_dim=num_tasks,
                                                 rank=1)  # The RBF is the one that has 2 inputs, Coregionalized works on the output-index dimension
m0 = GPy.models.GPRegression(X, Y, kern)
m0.optimize()
m0.predict(X)

#
lcm1 = GPy.util.multioutput.ICM(input_dim=2, num_outputs=2, kernel=kern)
m1 = GPy.models.GPRegression(X, Y, lcm1)
m1.optimize()
m1.predict(X)

lcm2 = GPy.util.multioutput.ICM(input_dim=2, num_outputs=2, kernel=GPy.kern.RBF(2))

X_list = [X[X[:,2]==i, :2] for i in range(2)]
Y_list = [Y[X[:,2]==i] for i in range(2)]
m2 = GPy.models.GPCoregionalizedRegression(X_list, Y_list, kernel=lcm2)
m2.optimize()

new_X = X  # In this case the input is not a list, but an array and it includes the third column with the index
m2.predict(new_X,Y_metadata={'output_index':np.ones((new_X.shape[0], 1)).astype(int)})  # Predicitions need us to tell which noise model we are using (again an output index per point)
m2.predict(new_X,Y_metadata={'output_index':np.zeros((new_X.shape[0], 1)).astype(int)})  # the output_index doesn't need to be all zeros or ones, it can be a combination. But it has to be a combination of zeros or ones (ie the outputs used to train the model).

