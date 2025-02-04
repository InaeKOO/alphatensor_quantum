# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate


from absl import app
import jax
import jax.numpy as jnp

from alphatensor_quantum.src.compiler import agent as agent_lib
from alphatensor_quantum.src.compiler import compile_config
from alphatensor_quantum.src.compiler.matrix_to_circuit import matrix_to_circuit


def main(_):
  # Set up the hyperparameters for the demo.
  config = compile_config.get_compile_config(
      use_gadgets=True  # Set to `False` for an experiment without gadgets.
  )
  exp_config = config.exp_config
  
  # Example matrix
  unitary_matrix = (1/np.sqrt(2)) * np.array([
    [1, -1j, 0, 0],
    [-1j, 1, 0, 0],
    [0, 0, 1, -1j],
    [0, 0, -1j, 1]
  ])

  # Convert matrix to circuit
  circuit = matrix_to_circuit(unitary_matrix)
  print(circuit.draw)
  
  # Initialize the agent and the run state.
  agent = agent_lib.Agent(config)
  run_state = agent.init_run_state(jax.random.PRNGKey(2024))

  # Main loop.
  previous_avg_return = None
  for step in range(
      0, exp_config.num_training_steps, exp_config.eval_frequency_steps
  ):
    time_start = time.time()
    run_state = agent.run_agent_env_interaction(step, run_state)
    time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
    # Keep track of the average return (for reporting purposes). We use a
    # debiased version of `avg_return` that only includes batch elements with at
    # least one completed episode.
    num_games = run_state.game_stats.num_games
    avg_return = run_state.game_stats.avg_return
    avg_return = jnp.sum(
        jnp.where(
            num_games > 0,
            avg_return / (1.0 - exp_config.avg_return_smoothing ** num_games),
            0.0
        ),
        axis=0
    ) / jnp.sum(num_games > 0, axis=0)

    print(
        f'Step: {step + exp_config.eval_frequency_steps} .. '
        f'Running Average Returns: {avg_return} .. '
        f'Time taken: {time_taken} seconds/step'
    )
    for t, target_circuit in enumerate(config.env_config.target_circuit_types):
      tcount = int(-run_state.game_stats.best_return[t])
      print(f'  Best T-count for {target_circuit.name.lower()}: {tcount}')

        # Check for convergence
    if previous_avg_return is not None:
        avg_return_change = jnp.abs(avg_return - previous_avg_return)
        if jnp.all(avg_return_change <= 0.01):
            print("Convergence reached. Ending training.")
            break

    previous_avg_return = avg_return


if __name__ == '__main__':
  app.run(main)
