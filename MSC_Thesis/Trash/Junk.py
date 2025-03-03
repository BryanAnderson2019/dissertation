import numpy as np
from numpy import expand_dims

#State space
random_array_1 = np.random.randint(0, 100, size=(13, 13))
random_array_2 = np.random.randint(0, 100, size=(13, 13))
random_array_3 = np.random.randint(0, 100, size=(13, 13))
array_13 = np.array([random_array_1,random_array_2,random_array_3])
#print(array_13)
#print(array_13.shape)


#Action space
shape = (12,)
array3 = np.random.randint(0, 100, size=shape)
array4 = np.random.randint(0, 100, size=shape)
array5 = np.random.randint(0, 100, size=shape)
array_12 = np.array([array3, array4,array5])
#print(array_12)
#print(array_12.shape)

#flattened
flattened_state_space = [array.flatten() for array in array_13]

main_array_state = np.empty((len(flattened_state_space),),dtype=object)

# Assign flattened arrays to main_array_state
for i in range(len(flattened_state_space)):
    main_array_state[i] = flattened_state_space[i]

print(main_array_state.shape)
print(main_array_state[0].shape)
print(main_array_state[0][0].shape)


flattened_action_space = [array.flatten() for array in array_12]
# Assign flattened arrays to main_array_state
main_action = np.empty((len(flattened_action_space),),dtype=object)
for i in range(len(flattened_action_space)):
    main_action[i] = flattened_action_space[i]

print(main_action.shape)
print(main_action[0].shape)
print(main_action[0][0].shape)

final_array = np.array((main_array_state, main_action))
print(final_array.dtype)

print(final_array[1])
