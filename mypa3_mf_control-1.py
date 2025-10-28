from typing import Any
import random
import gymnasium as gym

def argmax_action(d: dict[Any,float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    pass

class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi
    
    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        pass

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        pass

    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy
        If you want to use the eps_greedy, call the eps_greedy function in this function and return the action.

        Args:
            ss (int): state

        Returns:
            int: action
        """
        pass       
    
    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int,int,float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to be generated for evaluation. From the initial state, always take the greedily best action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        episode = list()
        done = False
        
        # TODO: implement here

        return (episode, done)

    def calc_return(self, episode: list[tuple[int,int,float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        pass   

class DoubleQLAgent(ValueRLAgent):  
    def __init__(self, env, gamma = 0.98, eps = 0.2, alpha = 0.02, total_epi = 5000):
        super().__init__(env, gamma, eps, alpha, total_epi)
        
    def learn(self):
        """Double Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """   
        pass
