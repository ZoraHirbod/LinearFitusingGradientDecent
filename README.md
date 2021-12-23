# LinearFitusingGradientDescent
Here, I apply the Gradient Descent method for line fitting (one of the simplest models). 
I start by explaining what the Gradient Descent is. Let’s consider a potential field function over a vector space and call the function f(x). The aim is to minimize or maximize the f(x) in the vector space. The gradient descent method idea for minimization is to take very small repeated steps and follow the opposite direction of the gradient. If travel is along the direction of the gradient, a higher value for f(x) is obtained. As a result, we keep updating the position of f(x) every time as follows:
x_(n+1)=x_n-α∇f(x_n)
The new value of x is built from the old value of x, and it subtracts from the value of step multiplied by the gradient value of the old x. We cannot go too far with choosing the steps since the gradient in each point is a localized indicator of how to change the f(x). Hence,  the step (alpha) should be a very small value. Alpha is known as a learning rate in ML. The NumPy library is used for linear fit using gradient descent. 

The medium article:https://zorahirbodvash.medium.com/linear-fit-using-gradient-descent-with-numpy-71d7a058eb6b
