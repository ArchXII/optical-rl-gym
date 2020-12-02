import gym
from typing import List

from optical_rl_gym.envs.power_aware_rmsa_env import SimpleMatrixObservation, PowerAwareRMSA
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.gnpy_utils import propagation

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 250
logging.getLogger('rmsacomplexenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 100

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))


def power_optimization_sapff(env: PowerAwareRMSA) -> List[int]:
    """
    Validation to find shortest available path. Finds the first fit, then determines the optimal power for that path.

    :param env: The environment of the simulator
    :return: action of iteration (path, spectrum resources, power)
    """
    power = -16  # Lower bound of possible powers minus one.
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        # print(path.best_modulation['minimum_osnr'])
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                # iteratively test initial power levels on the chosen path
                best_osnr = 0
                env.counter.add_min_osnr(path.best_modulation['minimum_osnr'])
                while power < 10:
                    power += 1
                    # Calls GNPy to calculate the OSNR of the transmission with the given path and initial power
                    osnr = np.mean(
                        propagation(power, env.gnpy_network, path.node_list, initial_slot, num_slots, env.eqpt_library))
                    if osnr > best_osnr:
                        best_osnr = osnr
                    # If this power creates an adequate OSNR, return it as part of the action
                    if osnr >= path.best_modulation['minimum_osnr']:
                        env.counter.add_osnr(osnr, 'success')
                        return [idp, initial_slot, power]
                # If no power in range gives a high enough OSNR, send the transmission with the minimum power.
                power = -16
                env.counter.add_osnr(best_osnr, 'failure')
                return [idp, initial_slot, power]
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources'], power]


init_env = gym.make('PowerAwareRMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_sapff, std_reward_sapff = evaluate_heuristic(env_rnd, power_optimization_sapff, n_eval_episodes=episodes)
print('PO-SAP-FF:'.ljust(8), f'{mean_reward_sapff:.4f}  {std_reward_sapff:>7.4f}')
print('Bit rate blocking:', (
        init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:',
      (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print('Throughput:', init_env.topology.graph['throughput'])
print(f'Total power: {10 * np.log10(init_env.total_power)} dBm')
print(f'Average power: {10 * np.log10(init_env.total_power / init_env.services_accepted)} dBm')
osnr_list = init_env.counter.get_osnr_list()
failed_osnr_list = []
failed_over_avg_min_list = []
succeeded_osnr_list = []
for i in osnr_list:
    if i[1] == 'failure':
        failed_osnr_list.append(i[0])
        if i[0] > np.mean(init_env.counter.get_min_osnr_list()):
            failed_over_avg_min_list.append(i[0])
    elif i[1] == 'success':
        succeeded_osnr_list.append(i[0])
print(f'Average minimum OSNR: {np.mean(init_env.counter.get_min_osnr_list())}')
print(f'Average successful OSNR: {np.mean(succeeded_osnr_list)}')
print(f'Average failed OSNR: {np.mean(failed_osnr_list)}')
print(f'Average failed OSNR greater than average minimum: {np.mean(failed_over_avg_min_list)}')
print(f'Number of failed OSNRs greater than average minimum: {len(failed_over_avg_min_list)}')
