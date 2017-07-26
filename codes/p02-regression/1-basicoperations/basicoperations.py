# Import CNTK library
import cntk
import numpy as np



#################################
#### Mathematical operations ####
#################################

# Initial definition
a = [1, 2, 3]
b = [3, 2, 1]

# Get the type of the variable
print(type(a))

# Subtraction
print(cntk.minus(a,b).eval())

# Additive
print(cntk.plus(a,b).eval())

# Element-wise division
print(cntk.element_divide(a,b).eval())


# Defining variable
variable = cntk.input_variable((2), np.float32)
print(variable)






