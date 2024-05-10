# use numpy to read the states from the zoo data, in the observation.txt file
# have the formulation
"""
1.25328269 -0.00537493 -0.00798181  0.00756171 -0.00609162 -0.00347085
  0.00222834  0.00149927  0.02725179 -0.07545965 -0.14656328 -0.78414866
  1.05276638 -0.33974209 -0.1567256   0.05242406 -0.02612962 
1.25236641e+00 -6.21155877e-03 -1.30869051e-02  1.36487917e-02
 -7.55431056e-03 -3.89047966e-03  2.70180669e-03  8.32246938e-04
  1.48235707e-02 -1.53610703e-01 -6.97183343e-02 -5.25131343e-01
  5.19558347e-01 -4.60747995e-02  4.25656850e-02  6.33788448e-02
 -1.36635129e-01 
....
"""

import numpy as np


data = np.zeros((1000, 17), dtype=np.float32)

# read the data from the file

data = np.loadtxt("research/zoo/observation.txt")

print(
    data.shape
)  # Should output (1000, 17) if there are 1000 lines exactly as described
print(data)  # This will print the array to verify it has loaded correctly
