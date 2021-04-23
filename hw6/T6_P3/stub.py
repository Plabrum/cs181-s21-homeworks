# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
# from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        
        # New values
        self.epsilon = 0.001
        # self.epsilon = 0.01
        self.alpha = 0.1
        self.gamma = 0.1
        self.explore_count = 0
        # End New Values

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def is_new(self):
        return np.array_equal(self.Q, np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE)))

    def decay_epsilon(self, epoch_num, tot_epochs):
        self.epsilon *= np.max(((tot_epochs-epoch_num)/ tot_epochs), 0)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.
        
        # Make adjustments if it's the first run - set position to 0,0 and randomly select option
        def q(s, a):
            return self.Q[a][s[0]][s[1]]
    
        def e_greedy_policy(s):
            explore = (np.random.rand() < self.epsilon)
            if explore:
                print("chose to explore! count:", self.explore_count)
                self.explore_count += 1

            # if equal choose randomly
            if q(s,0) == q(s,1):
                action = np.random.randint(2)
            else:
                action = (np.argmax([q(s,0), q(s,1)]) + explore) % 2
            return action

        if type(self.last_state) == type(None):
            s = (0,0)
            a = 0
            r = 0
        else:
            s = self.discretize_state(self.last_state)
            a = self.last_action
            r = self.last_reward
    
        s_prime = self.discretize_state(state)

        # Update Q function
        a_max = np.max([q(s_prime, a) for a in [0,1] ])

        self.Q[a][s[0]][s[1]] = q(s,a) + self.alpha*(r + self.gamma*a_max - q(s,a))

        a_prime = e_greedy_policy(s_prime)

        # decay epsilon
        self.epsilon *= 0.9

        self.last_action = a_prime
        self.last_state = state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        # learner.decay_epsilon(ii, iters)

        # Reset the state of the learner.
        learner.reset()
    # pg.quit()
    return


if __name__ == '__main__':

    def starter():
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games. You can update t_len to be smaller to run it faster.
        run_games(agent, hist, 100, 1)
        print("High score:", np.max(hist))
        print(agent.epsilon)
        print(hist)

        # Save history. 
        np.save('hist', np.array(hist))
    
    def find_best():
        test = [0.001, 0.01, 0.1, 0.2]
        best_test_score = 0
        best_params = (1,1,1)
        a_g_score = []
        for e in [0.001]:
            for a_count,a in enumerate(test):
                for g_count, g in enumerate(test):
                    pack_score = []
                    for i in range(2):
                        test_hist = []
                        test_agent = Learner()
                        test_agent.epsilon = e
                        test_agent.alpha = a
                        test_agent.gamma = g

                        run_games(test_agent, test_hist, 100, 1)
                        test_hist.sort(reverse=True)
                        score = np.mean(test_hist[:4])
                        
                        pack_score.append(score)
                    
                    top_score = np.mean(pack_score)
                    a_g_score.append([a_count, g_count, top_score])
                    print(f"Trying epsilon:{e}, alpha:{a}, gamma:{g}, with score:{top_score}")
                    if top_score > best_test_score:
                        best_test_score = top_score
                        best_params = (e,a,g)
        print("best parameters:", best_params, "best test score", best_test_score)
        np.save('a_g_score', np.array(a_g_score))
    # find_best()
    starter()
