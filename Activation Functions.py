#exp3
import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))
x_values=np.linspace(-10,10,200)
y_values=sigmoid(x_values)
import matplotlib.pyplot as plt
plt.plot(x_values,y_values,label="Sigmoid")
plt.title("Sigmoid activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

def tanh(x):
  return np.tanh(x)
y_values_tanh=tanh(x_values)
plt.plot(x_values,y_values_tanh,label='tanh',color='Green')
plt.title('tanh Activation function')
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

def relu(x):
  return np.maximum(0,x)
y_values_relu=relu(x_values)
plt.plot(x_values,y_values_relu,label='ReLU',color="Blue")
plt.title('ReLU Activation Function')
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

def leaky_relu(x,alpha=0.01):
  return np.where(x>0,x,alpha*x)
y_values_leaky_relu=leaky_relu(x_values)
plt.plot(x_values,y_values_leaky_relu,label="Leaky ReLU",color="Green")
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

def softmax(x):
  exp_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
  return exp_x/np.sum(exp_x,axis=-1,keepdims=True)
scores=np.array([4.0,2.0,0.3])
probs=softmax(scores)
print("Spftmax Probalities:",probs)