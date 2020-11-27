import gym
from optical_rl_gym.envs.power_aware_rmsa_env import shortest_path_first_fit, \
    shortest_available_path_first_fit_fixed_power, \
    least_loaded_path_first_fit, SimpleMatrixObservation, PowerAwareRMSA
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 250
logging.getLogger('rmsacomplexenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 50

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(f'../examples/topologies/germany50_eon_gnpy_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))


def sapfffs_electric_boogaloo(env: PowerAwareRMSA) -> int:
    """
    Validation to find shortest available path. Finds the first fit with a given fixed power.

    :param env: The environment of the simulator
    :return: action of iteration (path, spectrum resources, power)
    """
    power = 10   # Fixed power variable for validation method. Gets passed through simulator.
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                return [idp, initial_slot, power]
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources'], power]


# Random Policy
init_env = gym.make('PowerAwareRMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)

# Shortest Available Path First Fit Fixed Power
init_env = gym.make('PowerAwareRMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_sapfffp, std_reward_sapfffp = evaluate_heuristic(env_rnd, shortest_available_path_first_fit_fixed_power, n_eval_episodes=episodes)
print('SAP-FF-FP:'.ljust(8), f'{mean_reward_sapfffp:.4f}  {std_reward_sapfffp:>7.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print('Throughput:', init_env.topology.graph['throughput'])


init_env = gym.make('PowerAwareRMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_sapfffp, std_reward_sapfffp = evaluate_heuristic(env_rnd, sapfffs_electric_boogaloo, n_eval_episodes=episodes)
print('SAP-FF-FP:'.ljust(8), f'{mean_reward_sapfffp:.4f}  {std_reward_sapfffp:>7.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print('Throughput:', init_env.topology.graph['throughput'])


# env_sp = gym.make('PowerAwareRMSA-v0', **env_args)
# mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
# print('SP-FF:'.ljust(8), f'{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}')
# print('Bit rate blocking:', (env_sp.episode_bit_rate_requested - env_sp.episode_bit_rate_provisioned) / env_sp.episode_bit_rate_requested)
# print('Request blocking:', (env_sp.episode_services_processed - env_sp.episode_services_accepted) / env_sp.episode_services_processed)
#
# env_sap = gym.make('PowerAwareRMSA-v0', **env_args)
# mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
# print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
# print('Bit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
# print('Request blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
#
# env_llp = gym.make('PowerAwareRMSA-v0', **env_args)
# mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
# print('LLP-FF:'.ljust(8), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
# print('Bit rate blocking:', (env_llp.episode_bit_rate_requested - env_llp.episode_bit_rate_provisioned) / env_llp.episode_bit_rate_requested)
# print('Request blocking:', (env_llp.episode_services_processed - env_llp.episode_services_accepted) / env_llp.episode_services_processed)
