import sys
from abc import ABC, abstractmethod

class AgentState(ABC):
    def __init__(self, agent):
        self._agent = agent

        return
    
    @abstractmethod
    def choose_action(self, observation, from_target_actor=False):
        pass
    
    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def learn_step(self):
        pass



class TrainState(AgentState):
    def __init__(self, agent, replay_mem_ref):
        super().__init__(agent)

        self.replay_mem = replay_mem_ref

        return
    


    def choose_action(self, observation, from_target_actor=False):
        #print('TrainState choose_action called. rep mem size', self.replay_mem.mem_cntr)
        #sys.stdout.flush()
        return self._agent.choose_actor_action(observation, with_noise=True)



    def remember(self, state, action, reward, next_state, done):
        #print('TrainState remember called')
        #sys.stdout.flush()
        self.replay_mem.store_transition(state, action, reward, next_state, done)
        return
    


    def learn_step(self):
        #print('TrainState learn_step called')
        #sys.stdout.flush()
        return self._agent.learn()



class WarmupState(TrainState):
    def __init__(self, agent, curr_timestep, replay_mem_ref, warmup_steps):
        super().__init__(agent, replay_mem_ref)

        self.curr_timestep = curr_timestep
        self.warmup_steps  = warmup_steps

        return
    


    def choose_action(self, observation, from_target_actor=False):
        #print('WarmupState choose_action called. rep mem size', self.replay_mem.mem_cntr)
        #sys.stdout.flush()
        self.curr_timestep += 1

        if self.curr_timestep == self.warmup_steps:
            print('WARMUP ENDED')
            self._agent.change_state(TrainState(self._agent, self.replay_mem))

        return self._agent.choose_random_action(with_noise=False)
    


class EvalState(AgentState):
    def __init__(self, agent):
        super().__init__(agent)

        return
    

 
    def choose_action(self, observation, from_target_actor=False):
        #print('EvalState choose_action called')
        #sys.stdout.flush()
        # Choose an acion without any noise
        if from_target_actor:
            return self._agent.choose_actor_action_from_target(observation)
        
        return self._agent.choose_actor_action(observation, with_noise=False)



    def remember(self, state, action, reward, next_state, done):
        #print('EvalState remember called')
        #sys.stdout.flush()
        # Do nothing, we are in evaluation state, we do not 
        # store any experience
        return
    


    def learn_step(self):
        #print('EvalState learn_step called')
        #sys.stdout.flush()
        # Do nothing, we are in eval mode
        return (None, None, None)



class ExperienceGainState(AgentState):
    def __init__(self, agent, replay_mem_ref, steps, prev_state):
        super().__init__(agent)

        self.curr_step  = 0
        self.steps      = steps

        self.replay_mem = replay_mem_ref
        self.prev_state = prev_state

        return


    # Try choosing action with noise in order to get more states
    def choose_action(self, observation, from_target_actor=False):
        #print('ExperienceGainState choose_action called. rep mem size', self.replay_mem.mem_cntr)
        #sys.stdout.flush()
        #
        # Choose an acion without any noise because we want
        # to pick transitions that replicate the current NN agent
        return self._agent.choose_actor_action(observation, with_noise=True)



    def remember(self, state, action, reward, next_state, done):
        #print('ExperienceGainState remember called')
        #sys.stdout.flush()
        self.replay_mem.store_transition(state, action, reward, next_state, done)
        return
    


    def learn_step(self):
        #print('ExperienceGainState learn_step called')
        #sys.stdout.flush()
        fit_mse, fit_mape = None, None
        self.curr_step += 1

        if self.curr_step == self.steps:
            self._agent.change_state(self.prev_state)
            
            if self.replay_mem.is_full():
                #print('Updating target model')
                fit_mse, fit_mape = self._agent.update_target_actor()

        return (None, fit_mse, fit_mape)
