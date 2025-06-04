# rl-laboratory
A codebase to support multiple reinforcement learning experiments. As of right now, OpenAI gym is being foregone for custom environments; however, the API for environments in gym will be used, so integration will be a future feature. 

To be implemented:
- The step function could have the following format to work better with
    gym: next_state, reward, done, _ -- line 203
- Line 229: episode reward defined cumulatively, design choice
    worth reconsidering