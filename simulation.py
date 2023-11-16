from __future__ import division
import configparser
import subprocess
import gc
import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.append('..')
from utils import safe_remove, create_dir

def compute_coeff(airfoil, reynolds=500000, mach=0, alpha=3, n_iter=200, tmp_dir='./tmp', timeout=10):
    create_dir(tmp_dir)
    
    gc.collect()
    safe_remove(f'{tmp_dir}/airfoil.log')
    
    fname = f'{tmp_dir}/airfoil.dat'
    np.savetxt(fname, airfoil)
    
    try:
        xfoil_executable = 'xfoil'
        cmd = [xfoil_executable]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.stdin.write(f'load {tmp_dir}/airfoil.dat\n')
        process.stdin.write('af\n')
        process.stdin.write('OPER\n')
        process.stdin.write(f'VISC {reynolds}\n')
        process.stdin.write(f'ITER {n_iter}\n')
        process.stdin.write(f'MACH {mach}\n')
        process.stdin.write('PACC\n')
        process.stdin.write(f'{tmp_dir}/airfoil.log\n')
        process.stdin.write('\n')
        process.stdin.write(f'ALFA {alpha}\n')
        process.stdin.write('\n')
        process.stdin.write('quit\n')
        process.stdin.close()
        
        process.wait(timeout=timeout)

        # Continue with processing the results
        res = np.loadtxt(f'{tmp_dir}/airfoil.log', skiprows=12)
        if len(res) == 9:
            CL, CD = res[1], res[2]
        else:
            CL, CD = -np.inf, np.inf
            
    except subprocess.TimeoutExpired:
        print(f'Timeout expired. XFOIL process took too long.')
        process.terminate()
        CL, CD = -np.inf, np.inf
    except subprocess.CalledProcessError as ex:
        print(f'subprocess error: {ex}')
        CL, CD = -np.inf, np.inf
    except Exception as ex:
        print(f'An unexpected error occurred: {ex}')
        CL, CD = -np.inf, np.inf
        
    safe_remove(f'{tmp_dir}/:00.bl')
    
    return CL, CD

# Rest of your code...


# Rest of your code...


def read_config(config_fname):
    
    # Airfoil operating conditions
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    reynolds = float(Config.get('OperatingConditions', 'Reynolds'))
    mach = float(Config.get('OperatingConditions', 'Mach'))
    alpha = float(Config.get('OperatingConditions', 'Alpha'))
    n_iter = int(Config.get('OperatingConditions', 'N_iter'))
    
    return reynolds, mach, alpha, n_iter

def detect_intersect(airfoil):
    # Get leading head
    lh_idx = np.argmin(airfoil[:,0])
    lh_x = airfoil[lh_idx, 0]
    # Get trailing head
    th_x = np.minimum(airfoil[0,0], airfoil[-1,0])
    # Interpolate
    f_up = interp1d(airfoil[:lh_idx+1,0], airfoil[:lh_idx+1,1])
    f_low = interp1d(airfoil[lh_idx:,0], airfoil[lh_idx:,1])
    xx = np.linspace(lh_x, th_x, num=1000)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    # Check if intersect or not
    if np.any(yy_up < yy_low):
        return True
    else:
        return False

def evaluate(airfoil, return_CL_CD=False, config_fname='op_conditions.ini', tmp_dir='./tmp'):

    reynolds, mach, alpha, n_iter = read_config(config_fname)
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter, tmp_dir)
    perf = CL/CD
    if perf < -50 or perf > 220:
        perf = np.nan
    
    if return_CL_CD:
        return perf, CL, CD
    else:
        return perf
    
    
if __name__ == "__main__":
    
#    airfoil = np.load('tmp/a18sm.npy')
#    airfoils = np.load('data/airfoil_interp.npy')
    airfoils = np.load('data/xs_train.npy')
    
    idx = np.random.choice(airfoils.shape[0])
    airfoil = airfoils[idx]
    
    # Read airfoil operating conditions from a config file
    config_fname = 'op_conditions.ini'
    reynolds, mach, alpha, n_iter = read_config(config_fname)
    
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter)
    print(CL/CD, CL, CD)
    print(np.load('data/ys_train.npy')[idx])
    
#    val = evaluate(airfoil, return_CL_CD=False)
#    print(val)
