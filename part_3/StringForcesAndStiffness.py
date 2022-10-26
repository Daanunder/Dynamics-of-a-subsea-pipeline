import numpy as np
def StringForcesAndStiffness(NodePos, EA, l0, *args):
    TENSION_ONLY = 1
    WARN = False

    # Calculate extension
    dx = NodePos[0][1] - NodePos[0][0]
    dy = NodePos[1][1] - NodePos[1][0]
    dl = np.sqrt(dx**2 + dy**2)
    e = (dl-l0)/l0

    # Calculate tension
    t = EA*e
    if TENSION_ONLY != 0 and t < 0:
        WARN = True
        t = 0

    # Calculate orentation angle (COS and SIN)
    COS = dx/dl
    SIN = dy/dl

    # Calculate vector with internal forces
    Fi = t*np.array([[-COS, -SIN], [COS, SIN]])

    # Calculate K (in local coordinates)
    K = 1/l0*np.array([[EA, 0, -EA, 0], 
                        [0, t, 0, -t], 
                        [-EA, 0, EA, 0], 
                        [0, -t, 0, t]])
    
    # Calculate transformation matrix
    T = np.array([[COS, -SIN, 0, 0], 
                    [SIN, COS, 0, 0], 
                    [0, 0, COS, -SIN], 
                    [0, 0, SIN, COS]])
    
    # Calculate stiffness matrix in global coordinates
    K = np.matmul(T, np.matmul(K, np.transpose(T)))
    return Fi, K, t, WARN