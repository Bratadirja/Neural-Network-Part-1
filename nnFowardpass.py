import numpy as np

def fowardpass(input, weight, bias):
    w_sum = np.dot(input,weight) + bias

    #Liniear Activaion f(x) = x
    act = w_sum

    return act

# Pre-Trained weights & Biases after Training
w = np.array([[2.99999928]])
b = np.array([1.99999976])

# Initialize Input Data
inputs = np.array([[7],[8],[9],[10]])

# Ouput of Output Layer
o_out = fowardpass(inputs, w, b)

print("Output Layer Output (Linier)")
print("=============================")
print(o_out,"\n")


