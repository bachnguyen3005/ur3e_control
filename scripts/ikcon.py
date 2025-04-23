import numpy as np
from scipy.optimize import minimize

def ikcon(robot, T, q0=None, **kwargs):
    """
    Inverse kinematics by optimization with joint limits
    
    Parameters:
    -----------
    robot : Robot object
        The robot model
    T : numpy.ndarray or list of numpy.ndarray
        The end-effector pose as a 4x4 homogeneous transformation matrix,
        or sequence of poses
    q0 : numpy.ndarray, optional
        Initial joint angles for optimization
    **kwargs : dict
        Additional arguments to pass to scipy.optimize.minimize
        
    Returns:
    --------
    qstar : numpy.ndarray
        Joint coordinates (Nx1) or for sequence of poses (MxN)
    error : numpy.ndarray
        Final value of the objective function (1x1) or (Mx1)
    exitflag : numpy.ndarray
        Status from the optimizer (1x1) or (Mx1)
    output : dict
        Struct with information from the optimizer
        
    Notes:
    ------
    - Requires scipy.optimize.minimize
    - Joint limits are considered in this solution
    - Can be used for robots with arbitrary degrees of freedom
    - In the case of multiple feasible solutions, the solution returned
      depends on the initial choice of q0
    - Works by minimizing the error between the forward kinematics of the
      joint angle solution and the end-effector frame as an optimization
    - The objective function (error) is described as:
          sumsqr( (inv(T) @ robot.fkine(q) - eye(4)) * omega )
    """
    
    # Convert single pose to list if necessary
    single_pose = False
    if T.ndim == 2:
        T = [T]
        single_pose = True
    
    # Number of poses
    T_sz = len(T)
    
    # Get number of joints
    n = robot.n
    
    # Initialize output variables
    qstar = np.zeros((T_sz, n))
    error = np.zeros(T_sz)
    exitflag = np.zeros(T_sz, dtype=int)
    
    # Default initial guess if not provided
    if q0 is None:
        q0 = np.zeros(n)
    
    # Get joint limits
    if hasattr(robot, 'qlim'):
        if isinstance(robot.qlim, np.ndarray) and robot.qlim.ndim == 2:
            lb = robot.qlim[0, :]
            ub = robot.qlim[1, :]
        else:
            # Handle alternative formats
            lb = np.array([robot.qlim[i][0] for i in range(n)])
            ub = np.array([robot.qlim[i][1] for i in range(n)])
    else:
        # Default to no limits if qlim not found
        lb = np.full(n, -np.pi)
        ub = np.full(n, np.pi)
    
    # Calculate a reach parameter similar to MATLAB version
    # In Python RTB, we need to estimate the reach differently
    
    # Default reach value to avoid division by zero
    reach = 1.0
    
    try:
        # Try to use links
        if hasattr(robot, 'links') and len(robot.links) > 0:
            # Calculate reach as sum of link lengths if available
            reach_sum = 0
            for link in robot.links:
                if hasattr(link, 'a'):
                    reach_sum += abs(link.a)
                if hasattr(link, 'd'):
                    reach_sum += abs(link.d)
            if reach_sum > 0:
                reach = reach_sum
        
        # If we couldn't calculate from links, try alternate methods
        if reach == 1.0:
            if hasattr(robot, 'reach'):
                # Use reach property if available
                reach_val = robot.reach()
                if reach_val > 0:
                    reach = reach_val
            
            # If still using default, try to estimate from kinematics
            if reach == 1.0:
                # Create a stretched configuration
                stretched = np.zeros(n)
                # Set joint values to stretch the arm
                for i in range(n):
                    stretched[i] = 0.0
                
                # Get end-effector position at stretched config
                T_stretch = robot.fkine(stretched)
                
                # Extract position
                if hasattr(T_stretch, 'A'):
                    p_stretch = T_stretch.A[0:3, 3]
                elif hasattr(T_stretch, 'array'):
                    p_stretch = T_stretch.array()[0:3, 3]
                elif hasattr(T_stretch, 't'):
                    p_stretch = T_stretch.t
                else:
                    p_stretch = T_stretch[0:3, 3]
                
                # Calculate distance from origin (base) to end-effector
                reach_est = np.linalg.norm(p_stretch)
                if reach_est > 0:
                    reach = reach_est
    except Exception as e:
        # Keep the default reach if anything fails
        print(f"Warning: couldn't calculate reach, using default. Error: {e}")
    
    # Ensure reach is not zero
    if reach <= 0:
        reach = 1.0
    
    print(f"Using reach value: {reach}")
    
    # Weight matrix for the objective function
    omega = np.diag([1, 1, 1, 3/reach])
    
    # Define sum of squares function
    def sumsqr(A):
        return np.sum(A**2)
    
    # Process each pose in the sequence
    for t in range(T_sz):
        # Define the objective function for this pose
        def objective(q):
            # Forward kinematics at the proposed joint angles
            Tq = robot.fkine(q)
            
            # Extract the 4x4 matrix from the result (handle different return types)
            if hasattr(Tq, 'A'):
                Tq = Tq.A
            elif hasattr(Tq, 'array'):
                Tq = Tq.array()
            elif hasattr(Tq, 't'):  # For newer versions that use SE3 type
                # Get the full matrix from t and R
                Tq_full = np.eye(4)
                Tq_full[:3, :3] = Tq.R
                Tq_full[:3, 3] = Tq.t
                Tq = Tq_full
            elif isinstance(Tq, np.ndarray):
                Tq = Tq
            else:
                # Try to convert to numpy array as a last resort
                Tq = np.array(Tq)
            
            # Compute the error matrix
            try:
                T_inv = np.linalg.inv(T[t])
                error_matrix = (T_inv @ Tq - np.eye(4)) @ omega
            except Exception as e:
                print(f"Error computing objective: {e}")
                # Return a large error value on failure
                return 1e10
            
            # Return the sum of squared errors
            return sumsqr(error_matrix)
        
        # Initialize solver result
        result = None
        
        # Set up the bounds for the optimizer
        bounds = [(lb[i], ub[i]) for i in range(n)]
        
        # Set up optimization options
        options = {'disp': False}  # Default is no display
        if 'options' in kwargs:
            options.update(kwargs.pop('options'))
        
        # Run the optimizer with error handling
        try:
            result = minimize(
                objective, 
                q0, 
                bounds=bounds, 
                method='SLSQP',  # Similar to MATLAB's active-set
                options=options,
                **kwargs
            )
        except Exception as e:
            print(f"Optimization error: {e}")
            # Create a placeholder result in case of failure
            from types import SimpleNamespace
            result = SimpleNamespace(
                x=q0,
                fun=float('inf'),
                status=-1,
                success=False,
                message=str(e)
            )
        
        # Store the results
        qstar[t, :] = result.x
        error[t] = result.fun
        exitflag[t] = result.status
        
        # Use this solution as the initial guess for the next pose
        q0 = result.x
    
    # Return single result for single pose
    if single_pose:
        qstar = qstar[0]
        error = error[0]
        exitflag = exitflag[0]
        output = result
    else:
        output = result  # Only return the last result object
    
    # Return based on requested outputs
    return qstar, error, exitflag, output

# Extend the Robot class with the ikcon method
def add_ikcon_to_robot(robot_class):
    """
    Adds the ikcon method to a Robot class
    
    Usage:
    ------
    # After importing your robot class, e.g., from roboticstoolbox.models import UR3
    from ikcon import add_ikcon_to_robot
    add_ikcon_to_robot(rtb.models.UR3)
    
    # Now you can use ikcon method on your robot instance
    ur3 = rtb.models.UR3()
    q = ur3.ikcon(T)
    """
    def ikcon_method(self, T, q0=None, **kwargs):
        return ikcon(self, T, q0, **kwargs)
    
    # Add the method to the class
    setattr(robot_class, 'ikcon', ikcon_method)