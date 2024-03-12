# # import numpy as np
# # from hyperparameter import *
# # from random import choice
# # from math import *
# #
# #
# # class Env:
# #     def __init__(self):
# #         self.block_flag_list = None
# #         self.veh_task_data_list = None
# #         self.task_data_size_range = None
# #         self.loc_veh_list = None
# #         self.mec_remaining_cal_cap = None
# #         self.state = None
# #         self.M = 10  # vehicle num
# #         # constant
# #         self.p_uplink = 0.1
# #         self.bandwidth_nums = 600
# #         self.B = self.bandwidth_nums * 10 ** 6
# #         self.alpha0 = 1e-5
# #         self.p_noisy_nlos = 10 ** (-11)
# #         self.loc_mec = [500, 500, 100]
# #         self.p_noisy_los = 10 ** (-13)
# #         self.f_MEC = 5e10
# #         self.cpu_cycles = 1000
# #         self.f_veh = 5e10
# #         self.ground_width = 1000
# #         self.v = 10
# #         # state
# #         self.prepare()
# #
# #     def prepare(self):
# #         self.mec_remaining_cal_cap = Max_MEC_Cal_Cap
# #         self.loc_veh_list = np.random.randint(0, self.ground_width + 1, size=[self.M, 2])
# #         # self.task_data_size_range = choice([(1 * MB, 2 * MB), (20 * MB, 30 * MB), (200 * MB, 250 * MB)])
# #         self.task_data_size_range = (1 * MB, 1000 * MB)
# #         # self.veh_task_data_list = np.random.randint(*self.task_data_size_range, self.M)
# #         self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)
# #         # self.block_flag_list = np.random.randint(0, 2, self.M)
# #         self.get_state()
# #
# #     def time_delay(self, loc_veh, offloading_ratio, task_size, block_flag):
# #         dx = abs(loc_veh[0] - self.loc_mec[0])
# #         dy = abs(loc_veh[1] - self.loc_mec[1])
# #         dh = self.loc_mec[2]
# #         dist_mec_vehicle = dx * dx + dy * dy + dh * dh
# #         p_noise = self.p_noisy_los
# #         if block_flag == 1:
# #             p_noise = self.p_noisy_nlos
# #         g_mec_vehicle = abs(self.alpha0 / dist_mec_vehicle)
# #         trans_rate = self.B * log2(1 + self.p_uplink * g_mec_vehicle / p_noise)
# #         t_trans = offloading_ratio * task_size / trans_rate
# #         t_edge = offloading_ratio * task_size * self.cpu_cycles / self.f_MEC
# #         t_local = (1 - offloading_ratio) * task_size * self.cpu_cycles / self.f_veh
# #         if t_trans < 0 or t_edge < 0 or t_local < 0:
# #             raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
# #         time_delay = max([t_trans + t_edge, t_local])
# #         return t_local, t_trans, t_edge, time_delay
# #
# #     def get_state(self):
# #         self.state = np.append(self.mec_remaining_cal_cap / Max_MEC_Cal_Cap, self.loc_veh_list / self.ground_width)
# #         self.state = np.append(self.state, self.veh_task_data_list / max(self.veh_task_data_list))
# #         # self.state = np.append(self.state, self.block_flag_list)
# #         return self.state
# #
# #     def reset(self):
# #         self.prepare()
# #         return self.state
# #
# #     def reset_loc(self, time_decay):
# #         # update the location of all veh
# #         for i in range(self.M):
# #             tmp = np.random.rand(2)
# #             theta_veh = tmp[0] * 2 * np.pi
# #             dis_veh = tmp[1] * time_decay * self.v  # v is 10
# #             self.loc_veh_list[i][0] += cos(theta_veh) * dis_veh
# #             self.loc_veh_list[i][1] += sin(theta_veh) * dis_veh
# #             self.loc_veh_list[i] = np.clip(self.loc_veh_list[i], 0, self.ground_width)
# #
# #     def step(self, action):
# #
# #         def tanh_to_int(tanh_output, M):
# #             # Rescale tanh output to [0, M]
# #             rescaled_output = (tanh_output + 1) / 2 * M
# #             # Round to the nearest integer
# #             rounded_output = np.round(rescaled_output)
# #             # Clip to ensure the output is within [0, M]
# #             clipped_output = np.clip(rounded_output, 0, M)
# #
# #             return clipped_output.astype(int)
# #
# #         def tanh_to_float(tanh_output):
# #             # Rescale tanh output to [0, 1]
# #             rescaled_output = (tanh_output + 1) / 2
# #
# #             return rescaled_output
# #
# #         veh_id = tanh_to_int(action[0], self.M - 1)
# #         offloading_ratio = tanh_to_float(action[1])
# #         step_redo, terminate = False, False
# #         task_size = self.veh_task_data_list[veh_id]
# #         # block_flag = self.block_flag_list[veh_id]
# #
# #         min_cost1, min_cost2 = 100, 100
# #         for veh_id_i in range(self.M):
# #             block_flag = True
# #             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id_i], offloading_ratio,
# #                                                                    task_size, block_flag)
# #             p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
# #             energy = t_local * p1 + t_trans * p2 + t_edge * p3
# #             cost = time_delay * p + energy * q
# #             min_cost1 = min(min_cost1, cost)
# #
# #         for offloading_ratio_i in np.arange(0.1, 0.9 + 0.1, 0.1):
# #             block_flag = True
# #             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio_i,
# #                                                                    task_size, block_flag)
# #             p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
# #             energy = t_local * p1 + t_trans * p2 + t_edge * p3
# #             cost = time_delay * p + energy * q
# #             min_cost2 = min(min_cost2, cost)
# #
# #         if self.mec_remaining_cal_cap == 0:
# #             terminate = True
# #             cost = -100000
# #         elif self.mec_remaining_cal_cap - task_size < 0:
# #             self.veh_task_data_list[veh_id] = self.mec_remaining_cal_cap
# #             cost = -100
# #             step_redo = True
# #         else:
# #             block_flag = True
# #             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio,
# #                                                                    task_size, block_flag)
# #             if time_delay >= 10000:
# #                 cost = -100
# #                 step_redo = True
# #             else:
# #                 p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
# #                 energy = t_local * p1 + t_trans * p2 + t_edge * p3
# #                 cost = time_delay * p + energy * q
# #                 #cost = -cost
# #                 self.mec_remaining_cal_cap -= task_size
# #                 self.reset_loc(time_delay)
# #                 # reset the task of veh_id
# #                 # self.veh_task_data_list[veh_id] = np.random.randint(*self.task_data_size_range)
# #                 # reset the task of all
# #                 for veh_id in range(self.M):
# #                     # self.veh_task_data_list[veh_id] = np.random.randint(*self.task_data_size_range)
# #                     self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)
# #
# #         reward = - (((cost - min_cost1) / min_cost1) + (cost - min_cost2) / min_cost2)
# #         return self.get_state(), reward, terminate, step_redo
# import numpy as np
# from hyperparameter import *
# from random import choice
# from math import *
#
#
# class Env:
#     def __init__(self):
#         self.block_flag_list = None
#         self.veh_task_data_list = None
#         self.task_data_size_range = None
#         self.loc_veh_list = None
#         self.mec_remaining_cal_cap = None
#         self.state = None
#         self.M = 10  # vehicle num
#         # constant
#         self.p_uplink = 0.1
#         self.bandwidth_nums = 600
#         self.B = self.bandwidth_nums * 10 ** 6
#         self.alpha0 = 1e-5
#         self.p_noisy_nlos = 10 ** (-11)
#         self.loc_mec = [500, 500, 100]
#         self.p_noisy_los = 10 ** (-13)
#         self.f_MEC = 5e10
#         self.cpu_cycles = 1000
#         self.f_veh = 1e10
#         self.ground_width = 1000
#         self.v = 10
#         # state
#         self.prepare()
#
#     def prepare(self):
#         self.mec_remaining_cal_cap = MAX_MEC_COMPUTING
#         self.loc_veh_list = np.random.randint(0, self.ground_width + 1, size=[self.M, 2])
#         # self.task_data_size_range = choice([(1 * MB, 2 * MB), (20 * MB, 30 * MB), (200 * MB, 250 * MB)])
#         self.task_data_size_range = [1 * MB, 1000 * MB]
#         self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)
#         # self.block_flag_list = np.random.randint(0, 2, self.M)
#         self.get_state()
#
#     def time_delay(self, loc_veh, offloading_ratio, task_size, block_flag):
#         dx = abs(loc_veh[0] - self.loc_mec[0])
#         dy = abs(loc_veh[1] - self.loc_mec[1])
#         dh = self.loc_mec[2]
#         dist_mec_vehicle = dx * dx + dy * dy + dh * dh
#         p_noise = self.p_noisy_los
#         if block_flag == 1:
#             p_noise = self.p_noisy_nlos
#         g_mec_vehicle = abs(self.alpha0 / dist_mec_vehicle)
#         trans_rate = self.B * log2(1 + self.p_uplink * g_mec_vehicle / p_noise)
#         t_trans = offloading_ratio * task_size / trans_rate
#         t_edge = offloading_ratio * task_size * self.cpu_cycles / self.f_MEC
#         t_local = (1 - offloading_ratio) * task_size * self.cpu_cycles / self.f_veh
#         if t_trans < 0 or t_edge < 0 or t_local < 0:
#             raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
#         time_delay = max([t_trans + t_edge, t_local])
#         return t_local, t_trans, t_edge, time_delay
#
#     def get_state(self):
#         self.state = np.append(self.mec_remaining_cal_cap / MAX_MEC_COMPUTING, self.loc_veh_list / self.ground_width)
#         self.state = np.append(self.state, self.veh_task_data_list / np.max(self.veh_task_data_list))
#         # self.state = np.append(self.state, self.block_flag_list)
#         return self.state
#
#     def reset(self):
#         self.prepare()
#         return self.state
#
#     def reset_loc(self, time_decay):
#         # update the location of all veh
#         for i in range(self.M):
#             tmp = np.random.rand(2)
#             theta_veh = tmp[0] * 2 * np.pi
#             dis_veh = tmp[1] * time_decay * self.v  # v is 10
#             self.loc_veh_list[i][0] += cos(theta_veh) * dis_veh
#             self.loc_veh_list[i][1] += sin(theta_veh) * dis_veh
#             self.loc_veh_list[i] = np.clip(self.loc_veh_list[i], 0, self.ground_width)
#
#     def step(self, action):
#
#         def tanh_to_int(tanh_output, M):
#             # Rescale tanh output to [0, M]
#             rescaled_output = (tanh_output + 1) / 2 * M
#             # Round to the nearest integer
#             rounded_output = np.round(rescaled_output)
#             # Clip to ensure the output is within [0, M]
#             clipped_output = np.clip(rounded_output, 0, M)
#
#             return clipped_output.astype(int)
#
#         def tanh_to_float(tanh_output):
#             # Rescale tanh output to [0, 1]
#             rescaled_output = (tanh_output + 1) / 2
#
#             return rescaled_output
#
#         veh_id = tanh_to_int(action[0], self.M - 1)
#         offloading_ratio = tanh_to_float(action[1])
#         step_redo, terminate = False, False
#         task_size = self.veh_task_data_list[veh_id]
#         # block_flag = self.block_flag_list[veh_id]
#
#         min_cost1, min_cost2 = 100, 100
#         for veh_id_i in range(self.M):
#             block_flag = True
#             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id_i], offloading_ratio,
#                                                                    task_size, block_flag)
#             p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
#             energy = t_local * p1 + t_trans * p2 + t_edge * p3
#             cost = time_delay * p + energy * q
#             min_cost1 = min(min_cost1, cost)
#
#         for offloading_ratio_i in np.arange(0.1, 0.9 + 0.1, 0.1):
#             block_flag = True
#             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio_i,
#                                                                    task_size, block_flag)
#             p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
#             energy = t_local * p1 + t_trans * p2 + t_edge * p3
#             cost = time_delay * p + energy * q
#             min_cost2 = min(min_cost2, cost)
#
#         if self.mec_remaining_cal_cap == 0:
#             terminate = True
#             cost = 0
#         elif self.mec_remaining_cal_cap - task_size < 0:
#             self.veh_task_data_list[veh_id] = self.mec_remaining_cal_cap
#             cost = 0
#             step_redo = True
#         else:
#             block_flag = True
#             t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio,
#                                                                    task_size, block_flag)
#             if time_delay >= 10000:
#                 cost = 0
#                 step_redo = True
#             else:
#                 p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
#                 energy = t_local * p1 + t_trans * p2 + t_edge * p3
#                 cost = time_delay * p + energy * q
#                 # reward = -cost
#                 self.mec_remaining_cal_cap -= task_size
#                 self.reset_loc(time_delay)
#                 # reset the task of veh_id
#                 # self.veh_task_data_list[veh_id] = np.random.randint(*self.task_data_size_range)
#                 # reset the task of all
#                 for veh_id in range(self.M):
#                     self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)
#
#         reward = - (((cost - min_cost1) / min_cost1) + (cost - min_cost2) / min_cost2)
#         return self.get_state(), reward, terminate, step_redo
import numpy as np
from hyperparameter import *
from random import choice
from math import *


class Env:
    def __init__(self):
        self.block_flag_list = None
        self.veh_task_data_list = None
        self.task_data_size_range = None
        self.loc_veh_list = None
        self.mec_remaining_cal_cap = None
        self.state = None
        self.M = 10  # vehicle num
        # constant
        self.p_uplink = 0.1
        self.bandwidth_nums = 600
        self.B = self.bandwidth_nums * 10 ** 6
        self.alpha0 = 1e-5
        self.p_noisy_nlos = 10 ** (-11)
        self.loc_mec = [500, 500, 100]
        self.p_noisy_los = 10 ** (-13)
        self.f_MEC = 5e10
        self.cpu_cycles = 1000
        self.f_veh = 5e10
        self.ground_width = 1000
        self.v = 10
        # state
        self.prepare()

    def prepare(self):
        self.mec_remaining_cal_cap = Max_MEC_Cal_Cap
        self.loc_veh_list = np.random.randint(0, self.ground_width + 1, size=[self.M, 2])
        # self.task_data_size_range = choice([(1 * MB, 2 * MB), (20 * MB, 30 * MB), (200 * MB, 250 * MB)])
        self.task_data_size_range = (1 * MB, 1000 * MB)
        # self.veh_task_data_list = np.random.randint(*self.task_data_size_range, self.M)
        self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)
        # self.block_flag_list = np.random.randint(0, 2, self.M)
        self.get_state()

    def time_delay(self, loc_veh, offloading_ratio, task_size, block_flag):
        dx = abs(loc_veh[0] - self.loc_mec[0])
        dy = abs(loc_veh[1] - self.loc_mec[1])
        dh = self.loc_mec[2]
        dist_mec_vehicle = dx * dx + dy * dy + dh * dh
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_mec_vehicle = abs(self.alpha0 / dist_mec_vehicle)
        trans_rate = self.B * log2(1 + self.p_uplink * g_mec_vehicle / p_noise)
        t_trans = offloading_ratio * task_size / trans_rate
        t_edge = offloading_ratio * task_size * self.cpu_cycles / self.f_MEC
        t_local = (1 - offloading_ratio) * task_size * self.cpu_cycles / self.f_veh
        if t_trans < 0 or t_edge < 0 or t_local < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        time_delay = max([t_trans + t_edge, t_local])
        return t_local, t_trans, t_edge, time_delay

    def get_state(self):
        self.state = np.append(self.mec_remaining_cal_cap / Max_MEC_Cal_Cap, self.loc_veh_list / self.ground_width)
        self.state = np.append(self.state, self.veh_task_data_list / max(self.veh_task_data_list))
        # self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def reset(self):
        self.prepare()
        return self.state

    def reset_loc(self, time_decay):
        # update the location of all veh
        for i in range(self.M):
            tmp = np.random.rand(2)
            theta_veh = tmp[0] * 2 * np.pi
            dis_veh = tmp[1] * time_decay * self.v  # v is 10
            self.loc_veh_list[i][0] += cos(theta_veh) * dis_veh
            self.loc_veh_list[i][1] += sin(theta_veh) * dis_veh
            self.loc_veh_list[i] = np.clip(self.loc_veh_list[i], 0, self.ground_width)

    def step(self, action):

        def tanh_to_int(tanh_output, M):
            # Rescale tanh output to [0, M]
            rescaled_output = (tanh_output + 1) / 2 * M
            # Round to the nearest integer
            rounded_output = np.round(rescaled_output)
            # Clip to ensure the output is within [0, M]
            clipped_output = np.clip(rounded_output, 0, M)

            return clipped_output.astype(int)

        def tanh_to_float(tanh_output):
            # Rescale tanh output to [0, 1]
            rescaled_output = (tanh_output + 1) / 2

            return rescaled_output

        veh_id = tanh_to_int(action[0], self.M - 1)
        offloading_ratio = tanh_to_float(action[1])
        step_redo, terminate = False, False
        task_size = self.veh_task_data_list[veh_id]
        # block_flag = self.block_flag_list[veh_id]

        min_cost1, min_cost2 = 100, 100
        for veh_id_i in range(self.M):
            block_flag = True
            t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id_i], offloading_ratio,
                                                                   task_size, block_flag)
            p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
            energy = t_local * p1 + t_trans * p2 + t_edge * p3
            cost = time_delay * p + energy * q
            min_cost1 = min(min_cost1, cost)

        for offloading_ratio_i in np.arange(0, 1 + 0.1, 0.1):
            block_flag = True
            t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio_i,
                                                                   task_size, block_flag)
            p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
            energy = t_local * p1 + t_trans * p2 + t_edge * p3
            cost = time_delay * p + energy * q
            min_cost2 = min(min_cost2, cost)

        if self.mec_remaining_cal_cap == 0:
            terminate = True
            cost = -100000
        elif self.mec_remaining_cal_cap - task_size < 0:
            self.veh_task_data_list[veh_id] = self.mec_remaining_cal_cap
            cost = -100
            step_redo = True
        else:
            block_flag = True
            t_local, t_trans, t_edge, time_delay = self.time_delay(self.loc_veh_list[veh_id], offloading_ratio,
                                                                   task_size, block_flag)
            if time_delay >= 10000:
                cost = -100
                step_redo = True
            else:
                p1, p2, p3, p, q = 0.2, 0.3, 0.5, 0.5, 0.5
                energy = t_local * p1 + t_trans * p2 + t_edge * p3
                cost = time_delay * p + energy * q
                #cost = -cost
                self.mec_remaining_cal_cap -= task_size
                self.reset_loc(time_delay)
                # reset the task of veh_id
                # self.veh_task_data_list[veh_id] = np.random.randint(*self.task_data_size_range)
                # reset the task of all
                for veh_id in range(self.M):
                    # self.veh_task_data_list[veh_id] = np.random.randint(*self.task_data_size_range)
                    self.veh_task_data_list = np.random.randint(low=1 * MB, high=1000 * MB, size=self.M, dtype=np.int64)

        reward = - (((cost - min_cost1) / min_cost1) + (cost - min_cost2) / min_cost2)
        return self.get_state(), reward, terminate, step_redo
