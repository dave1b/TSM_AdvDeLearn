### Goal of Reinforcement Learning

Maximize the expected cumulative reward -> $\max \mathbb{E}[\sum_{t=0}^T r_t]$

### Notations

- $A_t$ -> action at time $t$
- $S_t$ -> state at time $t$
- $R_t$ -> reward at time $t$
- $G_t$ -> return at time $t$
- $\pi(s)$ -> policy
- $\pi(s)$ -> action $a$ taken in state $s$ under policy $\pi$
- $\pi(a|s)$ -> probability of taking action $a$ in state $s$ under policy $\pi$

### Value Function V(s)

The value function represents the expected cumulative reward (also called return) that an agent will receive from a
certain state $s$. In other words, it tells how good it is to be in a particular state.

$V(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R_{t+1} | S_0 = s]$

$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t=s]$

### Action-Value Function Q(s, a)

This function, also called the **Q-function**, represents the expected cumulative reward starting from a state $s$,
taking action $a$, and then following a policy. It tells how good it is to take action $a$ in state $s$.

$Q(s, a) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R_{t+1} | S_0 = s, A_0 = a]$


### Policy $\pi$(a,s)
A policy is a mapping from states to probabilities of selecting each possible action. It defines the agent's behavior.
There are two types of policies:
- **Deterministic policy**: Always selects the same action for a given state.
- **Stochastic policy**: Chooses actions according to a probability distribution over actions given a state.
  The goal of RL is to find the optimal policy $\pi^* $, which maximizes the cumulative reward.


---

### Markov Decision Process (MDP)

- Only the current state matters
- The future state is independent of the past states
- The environment is fully observable
- The environment is deterministic
- The environment is episodic
- The environment is finite
- Environment consists of states
- An action can be taken at each state
- The action will give reward and new state

-----
### Monte Carlo Methods
Estimate value functions by averaging the results of many random samples (trajectories) from an environment.
Value estimates are updated after an entire **episode** is completed.

Monte Carlo methods estimate the value function $V(s)$ or action-value function $Q(s,a)$ by averaging the returns (cumulative rewards) obtained after visiting a state $s$ or taking an action $a$.

**Key Features**:

- Learning from episodes: MC methods wait until an episode is over before calculating the return and updating value estimates. This works well in episodic tasks where the agent reaches a terminal state.

- Sample-based: They rely on sample episodes from actual experience in the environment, rather than requiring a full model of state transitions and rewards.

- Exploration requirement: To get a good estimate of the value function, the agent needs to explore the environment adequately (visit all relevant states and actions).


### Policy Iteration
    Used to find optimal policies in MDPs.

Policy iteration alternates between two main steps:

- Policy Evaluation: Given a policy ππ, estimate the value function VπVπ that describes the expected return starting from each state and following ππ.
- Policy Improvement: Use the value function VπVπ to improve the policy. The new policy π′π′ is determined by acting greedily with respect to the current value function, meaning that in each state, the agent chooses the action that maximizes the expected return.


### Temporal Difference Learning
TD learning allows an agent to update value estimates not only based on the reward at the end of an episode (like in Monte Carlo methods) but after every step, using the reward observed and the estimate of the next state’s value.



















