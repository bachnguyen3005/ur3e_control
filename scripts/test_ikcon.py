import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import transl, rpy2tr
from math import pi

# Import the custom ikcon implementation
from ikcon import ikcon

# Initial joint configuration
q0 = np.deg2rad(np.array([-180, -90, 0, -90, 0, 0]))
q_prefer = np.deg2rad(np.array([61.26, -81.48, -92.51, -91.86, 85.49, 6.96]))

# Create the robot
ur3 = rtb.models.UR3()
ur3.q = q0

# Target end-effector pose
Tep = transl(-0.145,-0.311,0.342) @ rpy2tr(0, pi/2, pi/2)

print('Target pose:')
print(Tep)

# Solve IK using our custom ikcon function
print("\nSolving with ikcon:")
sol, err, flag, out = ikcon(ur3, Tep, q0=q_prefer)
print(f"Joint solution (deg): {sol}")



