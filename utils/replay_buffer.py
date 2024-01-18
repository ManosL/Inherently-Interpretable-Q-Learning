import numpy as np



class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initializing memory buffers
        # each entry is like (state, action, next_state, reward, terminate)
        # Thus, I'll have 5 buffers, one for each coordinate
        self.state_memory      = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory     = np.zeros((self.mem_size, n_actions))
        self.reward_memory     = np.zeros(self.mem_size)
        self.terminal_memory   = np.zeros(self.mem_size, dtype=np.bool)

        return
    

    def empty_buffer(self):
        self.mem_cntr = 0

        return
    


    def is_full(self):
        return self.mem_cntr >= self.mem_size
    

    def len(self):
        return min(self.mem_cntr, self.mem_size)
    


    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index]        = state
        self.next_state_memory[index]   = next_state
        self.action_memory[index]       = action
        self.reward_memory[index]       = reward
        self.terminal_memory[index]     = done

        self.mem_cntr += 1

        return


    
    def sample_all(self):
        max_index     = min(self.mem_cntr, self.mem_size)
        batch_indexes = np.array(list(range(max_index)))

        states      = self.state_memory[batch_indexes]
        next_states = self.next_state_memory[batch_indexes]
        actions     = self.action_memory[batch_indexes]
        rewards     = self.reward_memory[batch_indexes]
        dones       = self.terminal_memory[batch_indexes]

        return states, actions, rewards, next_states, dones
    


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        if batch_size < max_mem:
            batch_indexes = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch_indexes = np.array(list(range(max_mem)))

        states      = self.state_memory[batch_indexes]
        next_states = self.next_state_memory[batch_indexes]
        actions     = self.action_memory[batch_indexes]
        rewards     = self.reward_memory[batch_indexes]
        dones       = self.terminal_memory[batch_indexes]

        return states, actions, rewards, next_states, dones



    def sample_buffer_most_recent(self, batch_size):
        if self.mem_cntr == 0:
            return None, None, None, None, None

        batch_indexes = []

        if self.mem_cntr <= batch_size:
            batch_indexes = np.array(list(range(self.mem_cntr)))
        else:
            last_index  = (self.mem_cntr % self.mem_size) - 1
            first_index = last_index - batch_size + 1

            if first_index >= 0:
                batch_indexes = np.array(list(range(first_index, last_index + 1)))
            else:
                batch_indexes  = list(range(0, last_index + 1))
                batch_indexes += list(range(self.mem_size + first_index, self.mem_size)) # first_index is negative 
                batch_indexes  = np.array(batch_indexes)

        assert(batch_indexes != [])

        if self.mem_cntr >= batch_size:
            assert((len(batch_indexes) == batch_size))

        states      = self.state_memory[batch_indexes]
        next_states = self.next_state_memory[batch_indexes]
        actions     = self.action_memory[batch_indexes]
        rewards     = self.reward_memory[batch_indexes]
        dones       = self.terminal_memory[batch_indexes]
      
        return states, actions, rewards, next_states, dones
    


    def sample_buffer_from_most_recent_window(self, batch_size, window_size):
        if window_size < batch_size or self.mem_cntr == 0:
            return None, None, None, None, None

        states, actions, rewards, next_states, dones = \
                                        self.sample_buffer_most_recent(window_size)
        
        if self.mem_cntr <= batch_size:
            return states, actions, rewards, next_states, dones

        num_samples = len(states)

        assert((num_samples > batch_size) and (num_samples <= window_size))

        batch_indexes = np.random.choice(num_samples, batch_size, replace=False)

        states      = self.state_memory[batch_indexes]
        next_states = self.next_state_memory[batch_indexes]
        actions     = self.action_memory[batch_indexes]
        rewards     = self.reward_memory[batch_indexes]
        dones       = self.terminal_memory[batch_indexes]

        return states, actions, rewards, next_states, dones
