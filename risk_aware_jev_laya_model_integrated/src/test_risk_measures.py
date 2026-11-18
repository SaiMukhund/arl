import numpy as np
from .risk_measures import upper_cvar,entropic_risk,entropic_premium
if __name__=="__main__":
    x=np.array([0,0,0,0,100.],dtype=float)
    assert upper_cvar(x,.2)==100.0
    assert entropic_risk(x,1.,100.)>=x.mean()
    assert entropic_premium(x,1.,100.)>=0.0
    print("Risk-measure tests passed.")
