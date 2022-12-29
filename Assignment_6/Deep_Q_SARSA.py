from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import random


class QLearningAgent:
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.5
        self.epsilon = 0.1
        self.playground_matrix = self.playground_representation()
        self.action_map = {0: "N", 1: "S", 2: "E", 3: "W"}
        self.direction_map = {0:'^', 1:'v', 2:'>', 3:'<'}

    def playground_representation(self):

        playground_matrix = np.zeros((9,9))
        walls_location = [(1,2), (1,3), (1,4), (1,5), (1,6), (2,6), (3,6), (4,6), (5,6), (7,1),(7,2),(7,3),(7,4)]
        snake_pit = [(6,5)]
        treasure = [(8,8)]

        for i in range(9):
            for j in range(9):
                if (i,j) in walls_location:
                    playground_matrix[i,j] = 0
                elif (i,j) in snake_pit:
                    playground_matrix[i,j] = -50
                elif (i,j) in treasure:
                    playground_matrix[i,j] = 50
                else:
                    playground_matrix[i,j] = -1

        return playground_matrix

    def plot_playground(self):


        #This will get plot the grid with 4 distinct colors
        # 0 - Black
        # -1 - Blue
        # -50 - Red
        # 50 - Green

        cmap = plt.cm.colors.ListedColormap(['red', 'grey', 'black' , 'green'])
        sns.heatmap(self.playground_matrix, annot=True, cmap=cmap)

        plt.show()
        plt.imshow(self.playground_matrix, cmap=cmap, vmin=-50, vmax=50)

        plt.show()


    def epsilon_greedy(self, Q, state, epsilon):
        #Choose A from S using policy derived from Q (epsilon-greedy)

        # Exploit
        action = np.argmax(Q[state[0], state[1], :])

        if np.random.uniform(0,1) < epsilon:
            #Explore
            action = np.random.choice([x for x in range(4)])

        return action

    def take_action(self, state, action):

        action = self.action_map[action]
        #Take action, observe r, s'
        if action == "N":
            #Move Up
            if state[0] == 0:
                s_prime = state
                r = -1
            else:
                s_prime = (state[0] - 1, state[1])

        elif action == "S":
            #Move Down
            if state[0] == 8:
                s_prime = state
                r = -1
            else:
                s_prime = (state[0] + 1, state[1])

        elif action == "E":
            #Move Right
            if state[1] == 8:
                s_prime = state
                r = -1
            else:
                s_prime = (state[0], state[1] + 1)

        elif action == "W":
            #Move Left
            if state[1] == 0:
                s_prime = state
                r = -1
            else:
                s_prime = (state[0], state[1] - 1)

        #Check if s' is a wall
        if self.playground_matrix[s_prime[0], s_prime[1]] == 0:
            s_prime = state
            r= -1

        if not s_prime == state:
            r = self.playground_matrix[s_prime[0], s_prime[1]]

        return s_prime, r

    def SARSA(self, alpha, gamma, epsilon, max_episodes, max_steps):
        Q = np.zeros((9,9,4))
        for episode in range(max_episodes):
            s = (0, 0)
            a = self.epsilon_greedy(Q, s, epsilon)

            for step in range(max_steps):
                s_prime, r = self.take_action(s, a)
                a_prime = self.epsilon_greedy(Q, s_prime, epsilon)
                Q[s[0], s[1], a] = Q[s[0], s[1], a] + alpha*(r + gamma*Q[s_prime[0], s_prime[1], a_prime] - Q[s[0], s[1], a])
                s = s_prime
                a = a_prime
                if r == 50 or r == -50:
                    break

        return Q

    def Q_learning(self, alpha, gamma, epsilon, max_episodes, max_steps):
        Q = np.zeros((9,9,4))

        for episode in range(max_episodes):
            s = (0, 0)
            for step in range(max_steps):
                a = self.epsilon_greedy(Q, s, epsilon)
                s_prime, r = self.take_action(s, a)
                Q[s[0], s[1], a] = Q[s[0], s[1], a] + alpha*(r + gamma*np.max(Q[s_prime[0], s_prime[1], :]) - Q[s[0], s[1], a])
                s = s_prime
                if r == 50 or r == -50:
                    break
        return Q


    def dyna_Q(self, alpha, gamma, epsilon, max_episodes, max_steps, n):

        #Dyna-Q
        Q = np.zeros((9,9,4))
        model = {}
        for episode in range(max_episodes):
            s = (0, 0)
            if episode % 100 == 0:
                print("Episode: ", episode)

            for step in range(max_steps):
                a = self.epsilon_greedy(Q, s, epsilon)
                s_prime, r = self.take_action(s, a)
                Q[s[0], s[1], a] = Q[s[0], s[1], a] + alpha*(r + gamma*np.max(Q[s_prime[0], s_prime[1], :]) - Q[s[0], s[1], a])
                model[(s, a)] = (s_prime, r)
                s = s_prime
                if r == 50 or r == -50:

                    break
            for i in range(n):
                s_ran, a_ran = random.choice(list(model.keys()))
                s_prime_ran, r_ran = model[(s_ran, a_ran)]
                Q[s_ran[0], s_ran[1], a_ran] = Q[s_ran[0], s_ran[1], a_ran] + alpha * (
                        r_ran + gamma * np.max(Q[s_prime_ran[0], s_prime_ran[1], :]) - Q[s_ran[0], s_ran[1], a_ran])

        return Q

    def Q_sanity_check(self, Q):

        print("Q_sanity_check")
        for i in range(9):
            for j in range(9):
                print(i,j,self.playground_matrix[i,j], Q[i,j,:])


    def best_Q(self, Q):

        best_action = np.argmax(Q, axis=2)
        best_action_dir = np.vectorize(self.direction_map.get)(best_action)
        for i in range(9):
            for j in range(9):
                if self.playground_matrix[i,j] == 0:
                    best_action_dir[i,j] = "X"
                elif self.playground_matrix[i,j] == -50:
                    best_action_dir[i,j] = "S"
                elif self.playground_matrix[i,j] == 50:
                    best_action_dir[i,j] = "T"

        print("C1",best_action_dir)


        cmap = plt.cm.colors.ListedColormap(['black', 'red', 'blue',
                                             'yellow','grey',"green"]) #,'grey' , 'white', 'green'])
        sns.heatmap(best_action,annot=best_action_dir, fmt="")
        #plt.imshow(best_action_dir)
        plt.show()



def main():
    agent = QLearningAgent()
    #agent.plot_playground()
    Q = agent.SARSA(agent.alpha, agent.gamma, agent.epsilon, 1000, 100)
    agent.Q_sanity_check(Q)
    agent.best_Q(Q)

    agent2 = QLearningAgent()
    Q2 = agent2.Q_learning(agent.alpha, agent.gamma, agent.epsilon, 1000, 100)
    agent2.Q_sanity_check(Q2)
    agent2.best_Q(Q2)

    agent3 = QLearningAgent()
    Q3 = agent3.dyna_Q(agent.alpha, agent.gamma, agent.epsilon, 1000, 100, 50)
    agent3.Q_sanity_check(Q3)
    agent3.best_Q(Q3)

if __name__ == "__main__":
    main()


