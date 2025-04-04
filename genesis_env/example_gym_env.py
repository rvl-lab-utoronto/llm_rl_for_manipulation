# just an example gym environment I made once so I can get a sense
# of stuff I need to remember to include 
import numpy as np
import gymnasium as gym

class BallLQR3D(gym.Env): 
    """A ball that moves with momentum that we want to send to the origin. 

    AKA the double-integrator in 3D. 
    """
    def __init__(
        self,
        random_seed = 42,
        context_type = 'goal', # two options: goal and dynamics. 
        dynamics_values = [10,10,10,0.99,0.99,0.99], # set == 1 for "no context" but each CMDP is a valid MDP as well
        destination_values = [0,0,0,0,0,0],
        rf_feature_size = 100,
        do_rf = False,
        observe_context = True,
        rf_random_seed = 0,
        timeout_length = 250):
        # stores stuff b/c it's useful
        self.state_size = 6
        self.action_size = 3
        self.generator = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.rf_size = rf_feature_size
        self.do_rf = do_rf
        self.observe_context = observe_context
        self.rf_random_seed = rf_random_seed
        self.timeout_length = timeout_length
        self.context_type = context_type

        
        if self.context_type == 'dynamics':
            self.context_values = np.array(dynamics_values)
            if observe_context: # to account for context 
                self.aug_state_size = self.state_size + 6 
        elif self.context_type == 'goal':
            self.context_values = np.array(destination_values)
            if observe_context: # to account for context 
                self.aug_state_size = self.state_size + 6
        else:
            print('ERROR: undetected context type')
            self.aug_state_size = self.state_size
        if not observe_context: # to account for context 
            self.aug_state_size = self.state_size 

        if do_rf:
            self.obs_size = self.rf_size
        else:
            self.obs_size = self.aug_state_size
        
        
        self.destination = destination_values
        self.dynamics = dynamics_values

        # generates dynamics/costs
        self.A, self.B, self.Q, self.R = self.get_ball_lqr_system()
        if self.do_rf:
            self.rf_mat = self.get_random_feature_matrix(self.aug_state_size,
                                                        rf_feature_size)
            
        # gymnasium stuff 
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(self.obs_size,),dtype=float)
        self.action_space = gym.spaces.Box(-np.inf,np.inf,shape=(3,),dtype=float)
        

    def get_ball_lqr_system(self):
        """ Generates a Contextual MDP where the context 
        is a the mass of the ball being manipulated. 

        See https://arxiv.org/pdf/1910.13614 for details 
        """
        ## notes on environment setup
        # we define state space as [z_x, z_y, v_x, v_y] where z is position, v is velocity
        # action space is [u_x, u_y] where u is force exerted on ball
        # cost is z_x^2 + z_y^2 + u_x^2 + u_y^2. so velocity isn't penalized, but
        # the stationary solution should be at zero velocity

        ## generates default dynamics/costs
        # creates initial matrices
        state_size = self.state_size
        action_size = self.action_size
        A = np.zeros(shape=(state_size,state_size)) 
        B = np.zeros(shape=(state_size,action_size))
        Q = np.zeros(shape=(state_size,state_size))
        R = np.zeros(shape=(action_size,action_size)) 
        # grabs relevant values from LQR's context
        c_set = self.dynamics
        m_x,m_y,m_z,k_x,k_y,k_z = c_set[0],c_set[1],c_set[2],c_set[3],c_set[4],c_set[5]
        # fills in values 
        A[0,0],A[0,3],A[1,1],A[1,4],A[2,2],A[2,5] = 1,1,1,1,1,1
        A[3,3],A[4,4],A[5,5] = k_x,k_y,k_z
        B[-1,-1],B[-2,-2],B[-3,-3] = 1/m_x, 1/m_y,1 /m_z
        Q[0:3,0:3] = 1
        #Q[0,0], Q[1,1], Q[2,2] = 1,1,1
        R[0,0],R[1,1],R[2,2] = 1,1,1
    


        return A,B,Q,R
    def get_random_feature_matrix(self,
                                  obs_size,
                                  rf_feature_size,
                                  sequentially_consistent = False):
        # NOTE: sequentially consistent is broken don't turn it on plz
        # :) 
        """We need to be able to project lower dimensional
        state into higher dimensions so we can induce a larger parameter
        count into our linear learner "artificially". this is how we do it
        """
        # Size stuff below for my convenience:
        #   rf      =      RM      @   obs
        # [|rf|,1] <- [|rf|,|obs|] x [|obs|,1]
        generator = np.random.default_rng(self.rf_random_seed)
        #rf_mat = generator.normal(size=(rf_feature_size,obs_size),scale=1/50)
        if sequentially_consistent:
            generator = np.random.default_rng(self.rf_random_seed)
            rf_mat = generator.normal(size=(rf_feature_size,obs_size),scale = 1)
        else:
            generator = np.random.default_rng(self.rf_random_seed+rf_feature_size)
            rf_mat = generator.normal(size=(rf_feature_size,obs_size),scale = 1)
        return np.array(rf_mat)
        
    def get_rf(self, obs,non_linear=True):
        rf = self.rf_mat @ obs
        if non_linear:
            #rf = rf * (rf > 0) # ReLU non-linearity
            #rf = 1/(1+np.exp(-rf)) # Sigmoid non-linearity
            rf = np.tanh(rf)
        return rf
    def step(self,
             action):
        # reminder of the Gym-style API
        # state, reward = env.step(action)
        # why no dones? not a terminating MDP

        # applies dynamics and cost
        state_mod = (self.state - self.destination)
        #print(state_mod)

        # applies dynamics and cost
        new_state = self.A @ self.state + self.B @ action
        #cost = (state_mod.T @ self.Q @ state_mod) + (action.T @ self.R @ action)
        cost = np.linalg.norm(state_mod) ** 2 

        # sets new state
        self.state = new_state

        # handles observation
        obs = self.state
        if self.observe_context:
            obs = np.concatenate((obs,self.encode_context(self.context_values)),axis=0)
        if self.do_rf:
            obs = self.get_rf(obs)
        # handles horizon timeout
        terminated = False
        if self.step_count >= self.timeout_length:
            terminated = True
            self.step_count = 0
        else:
            self.step_count += 1 

        return obs,-cost,terminated,False,{}
        #return obs,-cost,terminated,False,{}
    def reset(self, seed = None, options =  None):
        # not sure of a "principled" way of doing this. just going to 
        # follow the previous thoughts and just generate it randomly
        if seed is None:
            seed = self.random_seed
        # gym stuff 
        super().reset(seed=seed)
        generator = np.random.default_rng(self.random_seed)
        # resets timeout
        self.step_count = 0
        # handles start -> if goal-conditioned, starts at origin. If not, starts at random place on unit sphere 
        # both have small amount of starting velocity 
        if self.context_type == 'goal':
            state = (np.abs(np.sign(generator.normal(size=(self.state_size,)))) + generator.normal(size=(self.state_size,),scale=0.25))*0
            state[0:3] = state[0:3]*0
            state[-3:] = state[-3:]*0.2
        else:
            state = np.abs(np.sign(generator.normal(size=(self.state_size,)))) + generator.normal(size=(self.state_size,),scale=0.25)
            state[-3:] = state[-3:]*0.2
        self.state = np.copy(state)
        if self.observe_context:
            state = np.concatenate((state,self.encode_context(self.context_values)),axis=0)
        if self.do_rf:
            state = self.get_rf(state)
        return state, {}