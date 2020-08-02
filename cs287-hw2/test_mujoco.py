#import mujoco_py
#import gym
#env = gym.make('Hopper-v2')  # or 'Humanoid-v2' 
#
#s = env.reset()
#done = False
#i = 0
#while not done:
#	env.render()
#	i+=1
#env.close()
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gym.envs.mujoco import *
from envs.hopper_env import HopperModEnv
from envs.cheetah_env import CheetahModEnv
import numpy as np
import copy
import gym
from scipy.io import loadmat
from scipy.io import savemat
import moviepy.editor as mpy
from simulators import *
from rot_utils import *
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')

# implement the infinite horizon optimal feedback controller
def lqr_infinite_horizon(A, B, Q, R):
    """
    find the infinite horizon K and P through running LQR back-ups
    until l2-norm(K_new - K_curr, 2) <= 1e-4
    return: K, P
    """
    
    dx, du = A.shape[0], B.shape[1]
    P, K_current = np.eye(dx), np.zeros((du, dx))
    
    """YOUR CODE HERE"""
    # make a helper function
    def ATbA(A, b):
        return A.T.dot(b.dot(A))
    # calculate initial update
    K_new = -(np.linalg.inv(R + ATbA(B, P))).dot(B.T.dot(P.dot(A)))
    P = Q + ATbA(K_new, R) + ATbA(A+B.dot(K_new), P)
    tol = np.linalg.norm(K_new-K_current, ord=2)
    K_current = K_new
    # loop over
    while tol > 1e-4:
        K_new = -(np.linalg.inv(R + ATbA(B, P))).dot(B.T.dot(P.dot(A)))
        P = Q + ATbA(K_new, R) + ATbA(A+B.dot(K_new), P)
        tol = np.linalg.norm(K_new-K_current, ord=2)
        K_current = K_new
    
    """YOUR CODE ENDS HERE"""
    return K_new, P

# implement linearization about a point
def linearize_dynamics(f, x_ref, u_ref, dt, my_eps, x_ref_tplus1=None):
    """
    f : dynamics simulator
    my_eps : delta for forward and backward differences you'll need
    note: please use centered finite differences!
    
    x(:,t+1) - x_ref  approximately = A*( x(:,t)-x_ref ) + B* ( u(:,t) - u_ref ) + c
    If we pick x_ref and u_ref to constitute a fixed point, then c == 0 
    
    For part (b), you do not need to use the optional argument (nor c).
    For part (d), you'll have to revisit and modify this function 
        --at this point, you'll want to use the optional argument and the resulting c. 
    
    return: A, B, c
    """
    
    if x_ref_tplus1 is not None:
        x_ref_next = x_ref_tplus1
    else:
        x_ref_next = x_ref
    
    dx, du = x_ref.shape[0], u_ref.shape[0]
    A, B = np.zeros((dx, dx)).astype(np.float64), np.zeros((dx, du)).astype(np.float64)
    
    """YOUR CODE HERE"""
    for i in range(dx):
        for j in range(dx):
            xj_f = np.copy(x_ref)
            xj_b = np.copy(x_ref)
            xj_f[j] += my_eps/2
            xj_b[j] -= my_eps/2
    #             print(xj_f, xj_b)
            A[i, j] = (f(xj_f, u_ref, dt) - f(xj_b, u_ref, dt))[i]
    #             print((f(xj_f, u_ref, dt) - f(xj_b, u_ref, dt))[i])
    for i in range(dx):
        for j in range(du):
            uj_f = np.copy(u_ref)
            uj_b = np.copy(u_ref)
            uj_f[j] += my_eps/2
            uj_b[j] -= my_eps/2
    #             print(uj_f, uj_b)
            B[i, j] = (f(x_ref, uj_f, dt) - f(x_ref, uj_b, dt))[i]         
    #             print((f(x_ref, uj_f, dt) - f(x_ref, uj_b, dt))[i])
    A /= my_eps
    B /= my_eps
    #     print(A)
    #     print(B)
    """YOUR CODE ENDS HERE"""
    
    c = f(x_ref, u_ref, dt) - x_ref_next
    if len(B.shape) == 1:
        return A, B.reshape(-1, 1), c
    return A, B, c

def lqr_nonlinear(config):
    env = config['env']
    env_name = config['exp_name']
    f = config['f']
    dt = 0.1 # we work with discrete time
    my_eps = 0.01 # finite difference for numerical differentiation
    
    # load in our reference points 
    x_ref, u_ref = config['x_ref'], config['u_ref']
    
    # linearize
    A, B, c = linearize_dynamics(f, x_ref, u_ref, dt, my_eps)
    dx, du = A.shape[0], B.shape[1]
    Q, R = np.eye(dx), np.eye(du)*2
    
    # solve for the linearized system
    K_inf, P_inf = lqr_infinite_horizon(A, B, Q, R) # you implemented in part (a)
    
    # recognize the simulation code from part (a)? modify it to use your controller at each timestep
    def simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, T, noise=None):
        for s in np.arange(n_starting_states):
            x, u = np.zeros((K_inf.shape[1], T+1)), np.zeros((K_inf.shape[0], T+1))
            x[:,0] = starting_states[:,s]
            for t in np.arange(T):
                """YOUR CODE HERE"""
                u[:,t] = K_inf.dot(x[:,t])
                """YOUR CODE ENDS HERE"""
                x[:,t+1] = f(x[:,t], u[:,t], dt)
                if "p_val" in config.keys():
                    perturbation_values = config["p_val"]
                    perturb = perturbation_values[t//(T//len(perturbation_values))]
                    x[:,t+1] = f(x[:,t], u[:,t], dt, rollout=True,perturb=perturb)
                if env is not None:
                    if t % 5 == 0:
                        plt.clf()
                        plt.axis('off')
                        plt.grid(b=None)
                        plt.imshow(env.render(mode='rgb_array', width=256, height=256))
                        plt.title("Perturbation Magnitude {}".format(perturb))

                        # if noise is not None:
                        #     plt.savefig(env_name +'/gifs/plot_noise_'+ str(perturb) + '_' + str(t) + '.png', dpi=300)
                        # else:
                        #     plt.savefig(env_name +'/gifs/plot_noise_'+ str(perturb) + '_' + str(t) + '.png', dpi=300)
                        
                        display.clear_output(wait=True)
                        display.display(plt.gcf())
                
                if noise is not None:
                    x[:,t+1] += noise[:,t]
            if env is not None:
                plt.clf()
        
            plt.plot(x.T[:-1], linewidth=.6)
            plt.plot(np.squeeze(u.T[:-1])/10.0, linewidth=.7, linestyle='--') # scaling for clarity
            if 'legend' in config.keys():
                config['legend'].append('u')
                plt.legend(config['legend'])
            else:
                legend_elements = [Line2D([0], [0], label='x'),Line2D([0], [0], linestyle='--', label='u')]
                plt.legend(handles=legend_elements)
            plt.xlabel('time')
            plt.title(config["exp_name"])

            # if noise is not None:
            #     plt.savefig(env_name + '/plot_noise_' + str(s) + '.png', dpi=300)
            # else:
            #     plt.savefig(env_name + '/plot_' + str(s) + '.png', dpi=300)
            
            plt.show()
        
    # now let's simulate and see what happens for a few different starting states
    starting_states = config['ss']
    n_starting_states = starting_states.shape[1]
    T = config['steps'] # simulating for T steps
    simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, T)
    if 'noise' in config.keys():
        # and now in the presence of noise
        noise_id = config['noise']
        noise_loaded = loadmat("mats/"+noise_id+".mat")[noise_id]
        simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, noise_loaded.shape[1], noise=noise_loaded)

env = HopperModEnv()
x_ref, u_ref = np.zeros(11), np.zeros(env.action_space.sample().shape[0])
hopper_config = {
    'env': env,
    'f': env.f_sim,
    'exp_name': "Perturbed-Hopper",
    'steps': 500,
    'x_ref': x_ref,
    'u_ref': u_ref,
    'ss':  np.array([[np.concatenate([env.init_qpos[1:],env.init_qvel])]]),
    'p_val': [0, .1, 1, 10]
}
lqr_nonlinear(hopper_config)

