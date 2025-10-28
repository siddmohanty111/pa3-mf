from typing import Any
import random
import gymnasium as gym
import matplotlib.pyplot as plt

def argmax_action(d: dict[Any,float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """

    max_value = max(d.values())    

    # list of keys with the maximum value
    keys = [key for key, value in d.items() if value == max_value]

    # return a random key among the keys with the maximum value
    return random.choice(keys)

class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.1, total_epi: int = 5_000) -> None:
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

        q_table = dict()

        for key in range(n_states):
            q_table[key] = dict()
            for a in range(n_actions):
                q_table[key][a] = init_val

        return q_table

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        
        action = None

        if exploration and random.random() < self.eps:
            # explore
            action = self.env.action_space.sample()
        else:
            # exploit based on the sum of the two Q tables by creating a new dictionary for this state
            action = argmax_action(self.q[state])  

        return action

    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy
        If you want to use the eps_greedy, call the eps_greedy function in this function and return the action.

        Args:
            ss (int): state

        Returns:
            int: action
        """

        return self.eps_greedy(ss, exploration=True)

    
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

        state, _ = self.env.reset()

        for i in range(max_steps):
            action = self.eps_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            episode.append((state, action, reward))

            if terminated or truncated:
                done = True
                break

            state = next_state    

        return (episode, done)

    def calc_return(self, episode: list[tuple[int,int,float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """

        total_return = 0.0

        for s, a, r in episode.__reversed__():
               total_return = r + self.gamma * total_return
        
        return total_return if done else None

class DoubleQLAgent(ValueRLAgent):
    
    def __init__(self, env, gamma = 0.98, eps = 0.2, alpha = 0.02, total_epi = 5000):
        super().__init__(env, gamma, eps, alpha, total_epi)
        self.q1 = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.q2 = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.reward_history = []

    # Override epsilon greedy from superclass
    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        
        action = None

        if exploration and random.random() < self.eps:
            # explore
            action = self.env.action_space.sample()
        else:
            # exploit based on the sum of the two Q tables by creating a new dictionary for this state
            q_sum = dict()
            for a in range(self.env.action_space.n):
                q_sum[a] = self.q1[state][a] + self.q2[state][a]
            action = argmax_action(q_sum)  

        return action

    def plot_rewards(self, sliding = False):
        """Plot the average reward every ten episodes."""
        avg_rewards = []

        if sliding: 
            for i in range(len(self.reward_history) - 10):
                avg = sum(r for r in self.reward_history[i:i+10] if r is not None) / 10
                avg_rewards.append(avg)
            plt.plot(range(len(avg_rewards)), avg_rewards)
        else:
            for i in range(0, len(self.reward_history), 10):
                avg = sum(r for r in self.reward_history[i:i+10] if r is not None) / 10
                avg_rewards.append(avg)
            plt.plot(range(0, len(self.reward_history), 10), avg_rewards)

        plt.xlabel('Episodes')
        plt.ylabel('Average Reward (per 10 episodes)')
        plt.title('Average Reward over Time')
        plt.show()

    def learn(self):
        """Double Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """

        for epi in range(self.total_epi):
            episode = list()
            goal = False
            # reset environment
            state, _ = self.env.reset()
            # loop until terminal state reached or trucation step limit
            limit = 500 # arbitrarily defined
            for step in range(limit):
                # choose action based on eps-greedy
                action = self.eps_greedy(state, exploration=True)
                # take action and observe next state and reward
                ss, r, terminated, truncated, info = self.env.step(action)
                # randomly choose to update q1 or q2
                if random.random() < 0.5:
                    # Q1 chosen
                    # argmax action from q1
                    argmax_q1 = argmax_action(self.q1[ss])
                    # use this to get q value from q2
                    q2_val = self.q2[ss][argmax_q1]
                    # calculate target
                    target = r + self.gamma * q2_val
                    # update q1
                    delta = self.alpha * (target - self.q1[state][action])
                    self.q1[state][action] += delta
                else:
                    # Q2 chosen
                    # argmax action from q2
                    argmax_q2 = argmax_action(self.q2[ss])
                    # use this to get q value from q1
                    q1_val = self.q1[ss][argmax_q2]
                    # calculate target
                    target = r + self.gamma * q1_val
                    # update q2
                    delta = self.alpha * (target - self.q2[state][action])
                    self.q2[state][action] += delta
                if terminated:
                    goal = True
                    break
                if truncated:
                    goal = False
                    break
                episode.append((state, action, r))
                state = ss

            # Print out progress and q tables every 100 episodes
            if (epi + 1) % 100 == 0:
                print(f"Episode {epi + 1}/{self.total_epi} completed.")
                print("Q1 Table:")
                for s in self.q1:
                    print(f"State {s}: {self.q1[s]}")
                print("Q2 Table:")
                for s in self.q2:
                    print(f"State {s}: {self.q2[s]}")

            self.reward_history.append(self.calc_return(episode, goal))

def main():

    env = gym.make("CliffWalking-v1")
    agent = DoubleQLAgent(env)
    agent.learn()

    # evaluate the learned policy
    episode, done = agent.best_run()

    print("Optimal Episode:")
    for step in episode:
        print(step)
    print("Reached Goal:", done)

    agent.plot_rewards(sliding=True)
    agent.plot_rewards(sliding=False)

if __name__ == "__main__":
    main()