import numpy as np
def StringDynamicForces(NodePos, NodeVel, EA, C, l0, *args):
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

    # Calculate velocity extension
    dvx = NodeVel[0][1] - NodeVel[0][0]
    dvy = NodeVel[1][1] - NodeVel[1][0]
    de = (dvx*COS + dvy*SIN)/l0  

    # Calculate the viscous force
    Fvisc = de*C

    # Calculate vector with internal forces
    Fi = (t + Fvisc)*np.array([[-COS, -SIN], [COS, SIN]])
    return Fi