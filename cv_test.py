# Import packages.
import cvxpy as cp
import numpy as np

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost), [x[:2]>=np.zeros((2,))])
#print(prob)
prob.solve()

# Print result.
#print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
#print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)

print("")
print("NEW")
prob = cp.Problem(cp.Minimize(cost), [x[:2]>=0])
#print(prob)
prob.solve(warm_start=False)

# Print result.
#print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)