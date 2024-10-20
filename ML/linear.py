from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt 

# input data
x =list(map(int,input("Enter the Values of X: ").split(" ")))
y =list(map(int,input("Enter the Values of Y: ").split(" ")))
n = len(x)

# calc mean
x_mean = sum(x)/n
y_mean = sum(y)/n

a = []
b = []

for i in range(n):
  a.append(x[i] - x_mean)
  b.append(y[i] - y_mean)

ab = [a[i] * b[i] for i in range(n)]
a_square = [ a[i] ** 2 for i in range(n)]
b_square = [ b[i] ** 2 for i in range(n)]

r = sum(ab)/sqrt(sum(a_square) * sum(b_square))
del_x = sqrt(sum(a_square))/sqrt(n-1)
del_y = sqrt(sum(b_square))/sqrt(n-1)

b1 = r * del_y/del_x
b0 = y_mean - b1 * x_mean

print("B0: ",b0, " B1: ", b1)
print("Equation: y =", b0, "+",b1, "x")

sns.scatterplot(x=x,y=y)

x_pred = [i for i in range(min(x), max(x) + 1)]
y_pred = [b0 + (b1 * i) for i in x_pred]

sns.lineplot(x = x_pred, y = y_pred, color = 'red')
plt.show() 