from copy import deepcopy
import numpy as np
import json


ACTION_TO_CLOUD = 0

ZERO_RES = 1e-6
MAX_EDGE_NUM = 10
        
class SDN_Env():
    def __init__(self, conf_file='config1.json', conf_name='SDN_Config1', w=1.0, fc=None, fe=None, edge_num=None):
        # Read configuration file
        config = json.load(open(conf_file, 'r'))
        param = config[conf_name]

        # Environment parameters
        self.dt = param['dt']
        self.Tmax = param['Tmax']
        self.edge_num_L = param['edge_num_L']
        self.edge_num_H = param['edge_num_H']
        self.cloud_num_L = param.get('cloud_num_L', 1)  # Đọc giá trị cloud_num_L từ tệp cấu hình, mặc định là 1 nếu không có
        self.cloud_num_H = param.get('cloud_num_H', 3)  # Đọc giá trị cloud_num_H từ tệp cấu hình, mặc định là 3 nếu không có

        self.user_num = param['user_num']
        self.possion_lamda = param['possion_lamda']
        
        # Task parameters
        self.task_size_L = param['task_size_L']
        self.task_size_H = param['task_size_H']
        self.wave_cycle = param['wave_cycle']
        self.wave_peak = param['wave_peak']
        
        # Cloud and Edge computing frequencies
        self.cloud_freq = param['cloud_cpu_freq']
        self.edge_freq = param['edge_cpu_freq']
        self.cloud_cpu_freq_peak = param['cloud_cpu_freq_peak']
        self.edge_cpu_freq_peak = param['edge_cpu_freq_peak']
        
        # User-specified frequencies (optional)
        self.fc = fc
        self.fe = fe
        self.edge_n = edge_num
            
        # Computing resource capacities
        self.cloud_C = param['cloud_C']
        self.edge_C = param['edge_C']
        self.cloud_k = param['cloud_k']
        self.edge_k = param['edge_k']
        
        # User distribution parameters
        self.cloud_user_dist_H = param['cloud_user_dist_H']
        self.cloud_user_dist_L = param['cloud_user_dist_L']
        self.edge_user_dist_H = param['edge_user_dist_H']
        self.edge_user_dist_L = param['edge_user_dist_L']

        # Data rate parameters
        self.cloud_off_datarate = param['cloud_off_datarate']
        self.edge_off_datarate = param['edge_off_datarate']
        

        self.edge_band_width = param['edge_band_width']  
        self.cloud_off_band_width = param['cloud_off_band_width'] 

        # Bandwidth capacity between user and edge server
        self.LC = np.random.uniform(param['min_bandwidth_capacity'], param['max_bandwidth_capacity'], size=(self.user_num, self.edge_num))

        # Reward parameter
        self.reward_alpha = param['reward_alpha']
        
        # Weight parameter
        self.w = w
        
        # Initialize the environment
        self.reset()  

    
    def reset(self):
        # Reset environment variables
        self.step_cnt = 0
        self.task_size = 0
        self.task_user_id = 0
        self.step_cloud_dtime = 0
        self.step_edge_dtime = 0
        self.step_link_utilisation = 0
        self.rew_t = 0
        self.rew_e = 0
        self.arrive_flag = False
        self.invalid_act_flag = False
        self.cloud_off_list = []
        self.cloud_exe_list = []
        self.edge_off_lists = []
        self.edge_exe_lists = []
        self.unassigned_task_list = []
        self.action = ACTION_TO_CLOUD

        # Randomly determine the number of edge servers
        self.edge_num = np.random.randint(self.edge_num_L, self.edge_num_H + 1)
        if self.edge_n:
            self.edge_num = self.edge_n
        # Randomly determine the number of cloud servers
        self.cloud_num = np.random.randint(self.cloud_num_L, self.cloud_num_H + 1)
        if self.cloud_n:
            self.cloud_num = self.cloud_n
        # Set the action space size based on the number of edge servers
        self.action_space = self.edge_num + self.cloud_num

        # Initialize finish time array for cloud and edge servers
        self.finish_time = np.array([0] * (self.edge_num + self.cloud_num))

        # Randomly determine cloud CPU frequency and edge CPU frequencies
        self.cloud_cpu_freqs = np.random.uniform(self.cloud_freq - self.cloud_cpu_freq_peak, self.cloud_freq + self.cloud_cpu_freq_peak, size=self.cloud_num)
        self.cloud_cpu_freq = self.fc if self.fc else self.cloud_cpu_freq
        self.edge_cpu_freq = [0] * self.edge_num

        # Bandwidth capacity between user and edge server
        self.LC = np.random.uniform(param['min_bandwidth_capacity'], param['max_bandwidth_capacity'], size=(self.user_num, self.edge_num))

        # Calculate task size exponential theta based on CPU frequencies and capacities
        self.task_size_exp_theta = 0  # Reset the value
        for i in range(self.edge_num):
            self.edge_cpu_freq[i] = np.random.uniform(self.edge_freq - self.edge_cpu_freq_peak, self.edge_freq + self.edge_cpu_freq_peak)
            self.edge_cpu_freq[i] = self.fe if self.fe else self.edge_cpu_freq[i]
            self.edge_off_lists.append([])
            self.edge_exe_lists.append([])
            self.task_size_exp_theta += self.edge_cpu_freq[i] / self.edge_C

        # Calculate task size exponential theta for cloud servers
        for i in range(self.cloud_num):
            self.cloud_cpu_freqs[i] = np.random.uniform(self.cloud_freq - self.cloud_cpu_freq_peak, self.cloud_freq + self.cloud_cpu_freq_peak)
            self.cloud_cpu_freqs[i] = self.fc if self.fc else self.cloud_cpu_freqs[i]
            self.cloud_off_lists.append([])
            self.cloud_exe_lists.append([])
            self.task_size_exp_theta += self.cloud_cpu_freqs[i] / self.cloud_C

        # Set environment flags and variables
        self.done = False
        self.reward_buff = []

        # Generate random user distances for cloud and edge servers
        for i in range(self.cloud_num):
            cloud_dist = np.random.uniform(self.cloud_user_dist_L, self.cloud_user_dist_H, size=(1, self.user_num))
            self.user_dist = np.concatenate((self.user_dist, cloud_dist), axis=0)
        for i in range(self.edge_num):
            edge_dist = np.random.uniform(self.edge_user_dist_L, self.edge_user_dist_H, size=(1, self.user_num))
            self.user_dist = np.concatenate((self.user_dist, edge_dist), axis=0)

        # Generate tasks for the environment
        self.generate_task()

        # Return the initial observation
        return self.get_obs()

               
    def step(self, actions):
        # Kiểm tra xem môi trường đã kết thúc chưa
        assert self.done == False, 'environment already output done'
        self.step_cnt += 1  # Tăng bước thời gian lên 1
        self.step_cloud_dtime = 0  # Đặt lại thời gian đám mây
        self.step_edge_dtime = 0  # Đặt lại thời gian nút edge
        self.step_link_utilisation = 0  # Đặt lại khả năng sử dụng liên kết 
        finished_task = []  # Danh sách công việc đã hoàn thành

        #####################################################
        # Assignment of tasks (Phân công công việc)
        if self.arrive_flag:
            assert actions <= self.edge_num and actions <= self.cloud_num and actions >= ACTION_TO_CLOUD, 'action not in the interval %d, %d' % (
            actions, self.edge_num)
            self.action = actions
            self.arrive_flag = False
            the_task = {}
            the_task['start_step'] = self.step_cnt
            the_task['user_id'] = self.task_user_id
            the_task['size'] = self.task_size
            the_task['remain'] = self.task_size
            the_task['off_time'] = 0
            the_task['wait_time'] = 0
            the_task['exe_time'] = 0
            the_task['off_link_utilisation'] = 0  
            the_task['exe_link_utilisation'] = 0  

        if actions == ACTION_TO_CLOUD:
            the_task['to'] = 0
            # Calculate non-offloading link utilization based on the provided formula
            off_link_utilisation = np.sum([self.cloud_off_lists[m]['size'] for m in range(len(self.cloud_off_list))]) / (
                    self.cloud_off_band_width * self.dt)
            the_task['off_link_utilisation'].append(off_link_utilisation)
            self.cloud_off_list.append(the_task)
        else:
            e = actions
            the_task['to'] = e
            # Calculate offloading link utilization based on the provided formula
            off_link_utilisation = np.sum([self.edge_off_lists[m][e - 1]['size'] for m in range(len(self.edge_off_lists[e - 1]))]) / (
                    self.edge_off_band_width[e - 1] * self.dt)
            the_task['off_link_utilisation'].append(off_link_utilisation)
            self.edge_off_lists[e - 1].append(the_task)

        self.rew_t, self.rew_lu = self.estimate_rew()

        #####################################################
        # Generate arriving tasks (Tạo công việc mới)
        self.generate_task()
        #####################################################
        # Cloud network (Mạng đám mây)
        # Advance the progress of task offloading and execution (Tiến triển tiến độ giải nhiệm công việc và thực hiện)
        for n in range(self.cloud_num):
            used_time = 0
            while (used_time < self.dt):
                off_estimate_time = []
                exe_estimate_time = []
                task_off_num = len(self.cloud_off_list)
                task_exe_num = len(self.cloud_exe_list)
                # Estimate offloading time (Ước lượng thời gian giải nhiệm)
                for i in range(task_off_num):
                    the_user = self.cloud_off_list[i]['user_id']
                    estimate_time = self.cloud_off_list[i]['remain'] / self.cloud_off_datarate[the_user]
                    off_estimate_time.append(estimate_time)
                # Estimate execution time (Ước lượng thời gian thực hiện)
                if task_exe_num > 0:
                    cloud_exe_rate = self.cloud_cpu_freq / (self.cloud_C * task_exe_num)
                for i in range(task_exe_num):
                    estimate_time = self.cloud_exe_list[i]['remain'] / cloud_exe_rate
                    exe_estimate_time.append(estimate_time)
                # Run (in the shortest time) (Chạy - thời gian ngắn nhất)
                if len(off_estimate_time) + len(exe_estimate_time) > 0:
                    min_time = min(off_estimate_time + exe_estimate_time)
                else:
                    min_time = self.dt

                run_time = min(self.dt - used_time, min_time)

                # Advance offloading (Tiến triển giải nhiệm)
                cloud_pre_exe_list = []
                retain_flag_off = np.ones(task_off_num, dtype=np.bool)
                for i in range(task_off_num):
                    the_user = self.cloud_off_list[i]['user_id']
                    self.cloud_off_list[i]['remain'] -= self.cloud_off_datarate[the_user] * run_time
                    self.cloud_off_list[i]['off_link_utilisation'] += (
                        np.sum([task['size'] for task in self.cloud_off_list]) *
                        self.cloud_off_list[i]['size']
                    ) / (self.LC[the_user] * self.dt)
                    if self.cloud_off_list[i]['remain'] <= ZERO_RES:
                        retain_flag_off[i] = False
                        the_task = deepcopy(self.cloud_off_list[i])
                        the_task['remain'] = self.cloud_off_list[i]['size']
                        cloud_pre_exe_list.append(the_task)
                pt = 0
                for i in range(task_off_num):
                    if retain_flag_off[i] == False:
                        self.cloud_off_list.pop(pt)
                    else:
                        pt += 1
                # Advance execution (Tiến triển thực hiện)
                if task_exe_num > 0:
                    cloud_exe_size = self.cloud_cpu_freq * run_time / (self.cloud_C * task_exe_num)
                    cloud_exe_link_utilisation = (
                        np.sum([task['size'] for task in self.cloud_exe_list]) *
                        cloud_exe_size
                    ) / (self.LC[0] * self.dt)
                retain_flag_exe = np.ones(task_exe_num, dtype=np.bool)
                for i in range(task_exe_num):
                    self.cloud_exe_list[i]['remain'] -= cloud_exe_size
                    self.cloud_exe_list[i]['exe_link_utilisation'] += cloud_exe_link_utilisation
                    self.cloud_exe_list[i]['exe_time'] += run_time
                    if self.cloud_exe_list[i]['remain'] <= ZERO_RES:
                        retain_flag_exe[i] = False
                pt = 0
                for i in range(task_exe_num):
                    if retain_flag_exe[i] == False:
                        self.cloud_exe_list.pop(pt)
                    else:
                        pt += 1
                self.cloud_exe_list = self.cloud_exe_list + cloud_pre_exe_list
                used_time += run_time
        #####################################################
        # Edge network (Mạng nút edge)
        for n in range(self.edge_num):
            # Advance the progress of task offloading and execution (Tiến triển tiến độ giải nhiệm công việc và thực hiện)
            used_time = 0
            while (used_time < self.dt):
                off_estimate_time = []
                exe_estimate_time = []
                task_off_num = len(self.edge_off_lists[n])
                task_exe_num = len(self.edge_exe_lists[n])
                # Estimate offloading time (Ước lượng thời gian giải nhiệm)
                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    estimate_time = self.edge_off_lists[n][i]['remain'] / self.edge_off_datarate[n, the_user]
                    off_estimate_time.append(estimate_time)
                # Estimate execution time (Ước lượng thời gian thực hiện)
                if task_exe_num > 0:
                    edge_exe_rate = self.edge_cpu_freq[n] / (self.edge_C * task_exe_num)
                for i in range(task_exe_num):
                    estimate_time = self.edge_exe_lists[n][i]['remain'] / edge_exe_rate
                    exe_estimate_time.append(estimate_time)
                # Run (in the shortest time) (Chạy - thời gian ngắn nhất)
                if len(off_estimate_time) + len(exe_estimate_time) > 0:
                    min_time = min(off_estimate_time + exe_estimate_time)
                else:
                    min_time = self.dt

                run_time = min(self.dt - used_time, min_time)

                # Advance offloading (Tiến triển giải nhiệm)
                edge_pre_exe_list = []
                retain_flag_off = np.ones(task_off_num, dtype=np.bool)
                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    self.edge_off_lists[n][i]['remain'] -= self.edge_off_datarate[n, the_user] * run_time
                    self.edge_off_lists[n][i]['off_link_utilisation'] += (self.edge_off_datarate[n, the_user] * run_time) / (
            self.edge_band_width * task_off_num)
                    self.edge_off_lists[n][i]['off_time'] += run_time
                    if self.edge_off_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_off[i] = False
                        the_task = deepcopy(self.edge_off_lists[n][i])
                        the_task['remain'] = self.edge_off_lists[n][i]['size']
                        edge_pre_exe_list.append(the_task)
                pt = 0
                for i in range(task_off_num):
                    if retain_flag_off[i] == False:
                        self.edge_off_lists[n].pop(pt)
                    else:
                        pt += 1
                # Advance execution (Tiến triển thực hiện)
                if task_exe_num > 0:
                    edge_exe_size = self.edge_cpu_freq[n] * run_time / (self.edge_C * task_exe_num)
                    # Calculate exe_link_utilisation using the provided formula
                    edge_exe_link_utilisation = (edge_exe_size * self.edge_exe_list[n]['exe_link_utilisation']) / (
                        self.LC[n] * self.dt
                    )
                retain_flag_exe = np.ones(task_exe_num, dtype=np.bool)
                for i in range(task_exe_num):
                    self.edge_exe_lists[n][i]['remain'] -= edge_exe_size
                    self.edge_exe_list[i]['exe_link_utilisation'] += edge_exe_link_utilisation                    
                    self.edge_exe_lists[n][i]['exe_time'] += run_time
                    if self.edge_exe_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_exe[i] = False
                pt = 0
                for i in range(task_exe_num):
                    if retain_flag_exe[i] == False:
                        self.edge_exe_lists[n].pop(pt)
                    else:
                        pt += 1
                self.edge_exe_lists[n] = self.edge_exe_lists[n] + edge_pre_exe_list
                used_time += run_time

        #####################################################
        # Done condition (Điều kiện kết thúc)
        if (self.step_cnt >= self.Tmax):
            self.done = True
        done = self.done

        #####################################################
        # Observation encoding (Mã hóa quan sát)
        obs = self.get_obs()

        #####################################################
        # Reward calculation (Tính toán thưởng)
        reward = self.get_reward(finished_task)

        #####################################################
        # Additional information (Thông tin bổ sung)
        info = {}
        return obs, reward, done, info
    
    def generate_task(self):
    #####################################################
    # Generate arriving tasks
        task_num = np.random.poisson(self.possion_lamda)  # Generate a random number of tasks based on a Poisson distribution
        for i in range(task_num):
            task = {}
            theta = self.task_size_exp_theta + self.wave_peak*np.sin(self.step_cnt*2*np.pi/self.wave_cycle)
            # Calculate the task size based on an exponential distribution with a dynamic parameter
            task_size = np.random.exponential(theta)
            # Clip the task size to be within specified bounds
            task['task_size'] = np.clip(task_size, self.task_size_L, self.task_size_H)
            # Assign a random user ID to the task
            task['task_user_id'] = np.random.randint(0, self.user_num)
            # Add the newly generated task to the list of unassigned tasks
            self.unassigned_task_list.append(task)
        
        if self.step_cnt < self.Tmax:
            if len(self.unassigned_task_list) > 0:
                # If there are unassigned tasks, set the arrive_flag to True and pick the first task from the list
                self.arrive_flag = True
                arrive_task = self.unassigned_task_list.pop(0)
                self.task_size = arrive_task['task_size']
                self.task_user_id = arrive_task['task_user_id']
            else:
                # If there are no unassigned tasks, set arrive_flag to True with a task size of 0 and a random user ID
                self.arrive_flag = True
                self.task_size = 0
                self.task_user_id = np.random.randint(0, self.user_num)


    
    def get_obs(self):
        obs = {}
        
        servers = []
        cloud = []
        cloud.append(1)
        cloud.append(self.cloud_cpu_freq/1e9)
        cloud.append(self.edge_num)
        cloud.append(self.task_size/1e6)
        cloud.append(1-self.done)
        cloud.append(self.cloud_off_datarate[self.task_user_id]/1e6/100)
        cloud.append(len(self.cloud_exe_list))
        task_exe_hist = np.zeros([60])
        n = 0
        for task in self.cloud_exe_list:
            task_feature = int(task['remain']/1e6)
            if task_feature>=60:
                task_feature = 59
            task_exe_hist[task_feature] += 1
        cloud = np.concatenate([np.array(cloud), task_exe_hist], axis=0)
        servers.append(cloud)
        
        for ii in range(self.edge_num):
            edge = []
            edge.append(1)
            edge.append(self.edge_cpu_freq[ii]/1e9)
            edge.append(self.edge_num)
            edge.append(self.task_size/1e6)
            edge.append(1-self.done)
            edge.append(self.edge_off_datarate[ii,self.task_user_id]/1e6/100)
            edge.append(len(self.edge_exe_lists[ii]))
            task_exe_hist = np.zeros([60])
            n = 0
            for task in self.edge_exe_lists[ii]:
                task_feature = int(task['remain']/1e6)
                if task_feature>=60:
                    task_feature = 59
                task_exe_hist[task_feature] += 1
            edge = np.concatenate([np.array(edge), task_exe_hist], axis=0)
            servers.append(edge)
        
        obs['servers'] = np.array(servers).swapaxes(0,1)
        
        re = obs['servers']
        return re
    
    def estimate_rew(self):
        remain_list = []
        if self.action == ACTION_TO_CLOUD:
            for task in self.cloud_exe_list:
                remain_list.append(task['remain'])
            computing_speed = self.cloud_cpu_freq/self.cloud_C
            offload_time = self.task_size/self.cloud_off_datarate[self.task_user_id] if self.task_size>0 else 0
        else:
            for task in self.edge_exe_lists[self.action-1]:
                remain_list.append(task['remain'])
            computing_speed = self.edge_cpu_freq[self.action-1]/self.edge_C
            offload_time = self.task_size/self.edge_off_datarate[self.action-1][self.task_user_id] if self.task_size>0 else 0

        remain_list = np.sort(remain_list)
        
        last_size = 0
        t2 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t2 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        last_size = 0
        t_norm = 0
        t1 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            use_t = (size-last_size)/current_speed
            if t_norm + use_t >= offload_time:
                t_cut = offload_time - t_norm
                t1 += (task_num-i)*t_cut
                t_norm = offload_time
                remain_list[i] -= t_cut*current_speed
                remain_list[i] = 0 if remain_list[i]<ZERO_RES else remain_list[i]
                remain_list = remain_list[i:]
                break
            else:
                t1 += (task_num-i)*(size-last_size)/current_speed
                t_norm += use_t
            last_size = size
        
        remain_list = remain_list.tolist()
        remain_list.append(self.task_size)
        remain_list = np.sort(remain_list)
        last_size = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t1 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        reward_dt = t1 - t2
        if self.task_size > 0:
            reward_dt = -reward_dt*0.01
            reward_dlu = -self.step_link_utilisation*50
        else:
            reward_dt = 0
            reward_dlu = 0
        
        return reward_dt, reward_dlu
    
    def get_reward(self, finished_task):
        reward_dt, reward_dlu = self.estimate_rew()
        reward = self.w * reward_dt + (1.0 - self.w) * reward_dlu
        return reward

    
    def rander(self):
        pass
    