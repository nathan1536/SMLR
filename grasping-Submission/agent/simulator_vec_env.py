"""
Class to enable vectorized environment which can contain multiple iiwa envs and communicate with Unity through gRPC.
Ideally you do not need to modify anything here and you should always use this environment as a wrapper of a single
iiwa environment in order to make it possible to communicate with unity and train RL algorithms which expect openAI gym
interface in SB3
"""
import ast
import base64
import json
import time

import subprocess
from subprocess import PIPE
import roslibpy
import zmq
import grpc

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from utils import service_pb2_grpc
from utils.service_pb2 import StepRequest

from torch.nn.modules.linear import Identity
import torchvision

try:
    from utils_advanced.manual_actions import configure_manual_settings_and_get_manual_function
except:
    pass

from torch import nn
import os
import cv2
import torch
from PIL import Image
import cv2
import sys
import base64
import torchvision.transforms as T
sys.path.append('../../../repos/r3m/r3m/')
sys.path.append('agent/mlp/')

from mlp_model import MLP_2
from mlp_model import MLP_1
from mlp_model import R3M_FT
from r3m import load_r3m


class CommandModel:

    def __init__(self, id, command, env_key, value):
        self.id = id
        self.environment = env_key
        self.command = command
        self.value = value
        

class SimulatorVecEnv(DummyVecEnv):
    _client = None

    def __init__(self, env_fns, config, manual_actions_dict, reward_dict, spaces=None ):
        """
        envs: list of environments to create
        """
        DummyVecEnv.__init__(self, env_fns)
        # self.env_process = subprocess.Popen(
        #     'J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6/Unity3D.exe '
        #     + config['command_line_params'] + " -pn "+ str(config["port_number"]),
        #     stdout=PIPE, stderr=PIPE, stdin=PIPE,
        #     cwd='J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6',
        #     shell=False)

        self.current_step = 0
        self.config = config
        self.reward_dict = reward_dict
        self.communication_type = config['communication_type']
        self.port_number = config['port_number']
        print("Port number: " + config["port_number"])
        self.ip_address = config['ip_address']
        print("Ip address: " + config["ip_address"])
        self.start = 0
        self.nenvs = len(env_fns)
        self.train_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        # self.validation_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        self.envs = self.train_envs
        print("Number of envs: " + str(len(self.envs)))
        #self.envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        
        #######Load R3M, R3M+MLP-1, VGG-16+MLP-2#########
        # self.mlp_weight_path = 'agent/mlp/mlp.pt'
        # self.r3m_weight_path = 'agent/mlp/model_best_r3m4.ckpt'
        self.mlp_2_weight_path = 'agent/mlp/epoch_50_vgg_mlp_2.pkl'
        self.mlp_1_weight_path = 'agent/mlp/epoch_50_mlp_1.pkl'
        self.device = 'cuda'
        self.input_size = 2048
        self.output_size = 9
        self.use_mlp = config['use_mlp']
        self.backbone = config['backbone']


        if self.use_mlp:

            self.model = self.load_mlp(self.input_size, self.output_size)

        else:

            self.model = self.load_r3m_('resnet50')  # input choices: resnet50, resnet18, resnet34




        # self.mlp_weight_path = 'agent/mlp/model_best_mlp4.ckpt'



        #########################

        # Initial position flag for the manipulator/robot after reseting. 1 means different than the default vertical position #
        if (config["initial_positions"] is None or np.count_nonzero(config["initial_positions"]) != 0) and config["random_initial_joint_positions"] == False:
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1

        if self.communication_type == 'ROS':
            # Connect to ROS server
            if SimulatorVecEnv._client is None:
                SimulatorVecEnv._client = roslibpy.Ros(host=self.ip_address, port=int(self.port_number))
                SimulatorVecEnv._client.run()
            self.service = roslibpy.Service(SimulatorVecEnv._client, '/step', 'rosapi/GetParam')
            self.request = roslibpy.ServiceRequest([['name', 'none'], ['default', 'none']])
        elif self.communication_type == 'ZMQ':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://127.0.0.1:" + str(self.port_number))
        elif self.communication_type == 'GRPC':
            self.channel = grpc.insecure_channel(self.ip_address + ":" + str(self.port_number))
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
        else:
            print("Please specify either ROS or ZMQ communication mode for this environment")

        ###################################################################
        # Manual/hard-coded actions to command at the end of the episode  #
        # Disabled by default. Refer to utils_advanced/                   #
        ###################################################################
        self.robotic_tool = config["robotic_tool"]
        if(manual_actions_dict is not None):
            self.manual = manual_actions_dict["manual"]
            self.manual_behaviour = manual_actions_dict["manual_behaviour"] # Behaviour to excecute, 'planar_grasping' (down/close/up) or 'close_gripper' (close)
            self.manual_rewards = manual_actions_dict["manual_rewards"]     # True/False -> whether to add rewards/penalties during manual actions

            # Set-up manual actions and return a function which will be called after the end of the agent episode #
            self.manual_func = configure_manual_settings_and_get_manual_function(self, manual_actions_dict)
        else:
            self.manual = False

        if(self.robotic_tool == "None"):
            print("The robot has no tool attached to the end-effector")
        elif(self.robotic_tool == "2_gripper"):
            print("The robot has a 2-finger gripper attached to the end-effector")
        elif(self.robotic_tool == "3_gripper"):
            print("The robot has a 3-finger gripper attached to the end-effector")
        elif(self.robotic_tool == "calibration_pin"):
            print("The robot has a calibration pin attached to the end-effector")
        else:
            print("The robot has no tool attached to the end-effector")
            
    def load_r3m_(self, backbone):
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        r3m = load_r3m(backbone) # resnet50, resnet18, resnet34
        # r3m = R3M_FT(r3m)
        # r3m.load_state_dict(torch.load(self.r3m_weight_path, map_location='cuda:0'))
        r3m.eval()
        r3m.to(device)
        
        return r3m
    
    # def load_mlp(self, in_channel, out_channel):
    #
    #     if torch.cuda.is_available():
    #         device = "cuda"
    #     else:
    #         device = "cpu"
    #
    #     mlp_model = MLP(in_channel, out_channel)
    #     mlp_model.load_state_dict(torch.load(self.mlp_weight_path, map_location='cuda:0'))
    #     mlp_model.eval()
    #     mlp_model.to(device)
    #
    #     return mlp_model

    def load_mlp(self, in_channel, out_channel):

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # network50 = torchvision.models.resnet50(pretrained=False)
        # network50.fc = Identity()
        # network50.to(device)
        # r3m = load_r3m('resnet50')
        # r3m.to(device)
        if self.backbone == 'r3m':
            backbone = self.load_r3m_('resnet50')
            mlp_model = MLP_1(in_channel, out_channel, features=backbone)
            ckpt = torch.load(self.mlp_1_weight_path)
            mlp_model.load_state_dict(ckpt['model_state_dict'])

        elif self.backbone == 'vgg16':
            backbone = torchvision.models.vgg16(pretrained=False)
            features_ = backbone.features
            mlp_model = MLP_2(in_channel, out_channel, features=features_)
            ckpt = torch.load(self.mlp_2_weight_path)
            mlp_model.load_state_dict(ckpt['model_state_dict'])

        else:
            print("Incorrect Backbone.")



        # network50.eval()
        mlp_model.eval()
        mlp_model.to(device)

        return mlp_model
               
    
    # function to apply transform to the numpy image of each env and convert it to device(cuda or cpu) 
    def apply_transform_to_images_r3m(self, image):
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        # transforms = T.Compose([T.Resize(256),
        #     T.CenterCrop(224),
        #     T.ToTensor()])

        # transforms = T.Compose([T.Resize(224),
        #                         T.CenterCrop(224),
        #                         T.ToTensor()])

        transforms = T.Compose([T.Resize((224, 224)),
                                T.ToTensor()])


        # transforms = T.Compose([T.Resize(256),
        #     T.CenterCrop(224),
        #     T.ToTensor(),
        #     #T.Normalize(mean=[0.7826753009163441, 0.7899872787901007, 0.9358119223454758], std=[0.2124363898663008, 0.20321791028453737, 0.14656470348637965])
        # ])

        preprocessed_image = transforms(image)
        preprocessed_image.to(device)
        
        return preprocessed_image
        

    def switch_to_training(self):
        self.envs = self.train_envs

    def switch_to_validation(self):
        self.envs = self.validation_envs

    def step(self, actions, dart_convert=True):
        """
           :param actions: list[list[float]]] list of lists. Each sublist entails actions that will be transfered to each corresponding env.
                           These actions originate either from a model-based controller or from the output of a Neural Network of a RL model

           :param dart_convert: bool, if set to True it means the actions are in task-space and must be converted with IK to the joint space. 
                                      The UNITY simulator always expects velocities in joint space.
        """

        self.current_step += 1

        if dart_convert and 'dart' in self.config['env_key']:
            actions_converted = []
            for env, action in zip(self.envs, actions):
                act = env.update_action(action)         # Convert agent action to UNITY format - joint space and tool action
                actions_converted.append(act)
            actions = actions_converted

        ####################################################################################################
        # Send actions to UNITY (JSON message) and update the envs with the new observation                #
        # after executing these actions (update the dart kinematic (robotic) chain, agents time-step, etc) #
        ####################################################################################################
        terminated_environments, rews, infos, observation_mlp, dones = self._send_actions_and_update(actions)

        # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
        if(self.config["simulation_mode"] == 'train'):
            self.render()

        ##########################################################################################################
        # TODO: Make sure the last observation for terminated environment is the correct one:                    #
        #       https://github.com/hill-a/stable-baselines/issues/400 talks about reset observation being wrong, #
        #       use terminated observation instead                                                               #
        # TODO: make sure this does not cause problems, when the when images are transfered it might be slow     #
        #       to get all environment observations again                                                        #   
        ##########################################################################################################

        # Reset all the terminated environments #
        if len(terminated_environments) > 0:

            ####################################################################################
            # Manual actions at the end of the agent episode.                                  #
            # Disabled by default. Refer to utils_advanced/                                    #
            # Note: the envs should terminate at the same timestep due to UNITY sychronization #
            #       for collided envs -> send zero velocities for the remaining steps          #
            ####################################################################################
            if(self.manual == True): 
                self.manual_func(self, rews, infos, self.manual_rewards)

            # Successful episode: give terminal reward # 
            for env in terminated_environments:
                if env.get_terminal_reward():
                    rews[env.id] += self.reward_dict["reward_terminal"]
                    infos[env.id]["success"] = True

            ###########################################
            # Reset gym envs and UNITY simulator envs #
            ###########################################
            [env.reset() for env in terminated_environments]
            observation_mlp = self._send_reset_and_update(terminated_environments, time_step_update=True)

            # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
            if(self.config["simulation_mode"] == 'train'):
                self.render()

            ########################################################################################
            # The manipulator resets to non-zero initial joint positions.                          #
            # Correct the UNITY observation and update the dart chain due to UNITY synchronization #
            # Important: Make sure the agent episodes terminate at the same time-step in this case #           
            ########################################################################################
            if(self.flag_zero_initial_positions == 1):
                observation_mlp = self._send_zero_vel_and_update(self.envs, True)

        return np.stack(observation_mlp), np.stack(rews), np.stack(dones), infos

    def step_wait(self):
        # only because VecFrameStack uses step_async to provide the actions, then step_wait to execute a step
        return self.step(self.actions)

    def _create_request(self, command, environments, actions=None):
        """
            Create request to send to the UNITY simulator
        """
        content = ''
        if command == "ACTION":
            for act, env in zip(actions, environments):
                # print("id: {}\t step: {}\t action = {}".format(env.id, env.ts, str(np.around(act, decimals=3))))

                act_json = json.dumps(CommandModel(env.id, "ACTION", "manipulator_environment", str(act.tolist())), default=lambda x: x.__dict__)
                content += (act_json + ",")

        elif command == "RESET":
            #print("Time: " + str(time.time() - self.start))
            self.start = time.time()
            for env in environments:
                reset_string = str(env.reset_state)
                act_json = json.dumps(CommandModel(env.id, "RESET", "manipulator_environment", reset_string), default=lambda x: x.__dict__)
                content += (act_json + ",")

        return '[' + content + ']'

    def _send_request(self, content):

        # "{\"Environment\":\"manipulator\",\"Action\":\"" + translated_action + "\"}"
        if self.communication_type == 'ROS':
            self.request['name'] = content
            return self._parse_result(self.service.call(self.request))
        elif self.communication_type == 'ZMQ':
            self.socket.send_string(content)
            response = self.socket.recv()
            return self._parse_result(response)
        else:
            reply = self.stub.step(StepRequest(data=content))
            return self._parse_result(reply.data)

    def _parse_result(self, result):
        if self.communication_type == 'ROS':
            return ast.literal_eval(result['value'])
        elif self.communication_type == 'ZMQ':
            return ast.literal_eval(result.decode("utf-8"))
        else:
            return ast.literal_eval(result)

    def reset(self, should_reset=True):
        if should_reset:
            [env.reset() for env in self.envs]

        # Reset UNITY environments and update the agent envs (dart chain) #
        self._send_reset_and_update(self.envs, time_step_update=False)

        # Correct dart chain #
        latent_vector_array = self._send_zero_vel_and_update(self.envs, True)

        return np.array(latent_vector_array)

    ###########
    # Helpers #
    ###########
    def _send_reset_and_update(self, envs, time_step_update=True):
        """
            send a reset JSON command and update the envs (time steps, observations, dart chains)

            :param envs: envs
            :param time_step_update: whether to update the timestep of the agents

            :return: obs_converted
        """

        # UNITY #
        request = self._create_request("RESET", envs)
        observations = self._send_request(request)
        
        images = self.get_image_from_simulator_obs(observations)
        
        
        observation = self.return_image_based_observation(images)

        # Agents #
        state, _, _, _ = self._update_envs(observations, time_step_update=time_step_update)

        if self.use_mlp:

            for i, env in enumerate(envs):
                mlp_joint_obs = observation[i, :7]
                observation[i, :7] = env.normalize_joints(mlp_joint_obs)
                observation[i, 7] = observation[i, 7] / 1.4
                observation[i, 8] = observation[i, 8] / 0.73

        return observation
     

    def _send_actions_and_update(self, actions):
        """
            send actions to the UNITY envs and update the agents envs with the returned observations (time steps, observations, dart chains)
            Note: send zero velocities to collided envs

            :param actions: agent actions (UNITY format) for each env to be send to UNITY simulator

            :return: terminated_environments, rews, infos, obs_converted, dones 
        """

        rews = [0] * len(self.envs)
        if(self.config['env_key'] == 'iiwa_joint_vel'):
            action_dim = self.config['num_joints'] + 1 # For gripper
        elif(self.config['robotic_tool'].find("gripper") == -1):
            action_dim = 7
        else:
            action_dim = 8

        # For collided envs send zero velocities #
        for env in self.envs:
            if env.collided_env == 1:
                actions[env.id] = np.zeros(action_dim)

        #print("current step:" + str(self.current_step))
        # create request containing all environments with the actions to be executed

        #####################################################################################################
        # Execute a UNITY simulation step for all environments and parse the returned observations (result) #
        #####################################################################################################
        request = self._create_request("ACTION", self.envs, actions)
        observations = self._send_request(request)

        ##########################################################################################
        # Update the envs using the obs returned from UNITY (dart chain of the manipulator, etc) #
        ##########################################################################################
        observations_converted = []
        terminated_environments = []                                                             # Assume done in the same timestep else move it outside the for
        dones = []
        infos = []
        
        images = self.get_image_from_simulator_obs(observations)
        
        
        observation = self.return_image_based_observation(images)

        state, rews, dones, infos = self._update_envs(observations, time_step_update=True)

        if self.use_mlp:

            for i, env in enumerate(self.envs):
                mlp_joint_obs = observation[i, :7]
                observation[i, :7] = env.normalize_joints(mlp_joint_obs)
                observation[i, 7] = observation[i, 7] / 1.4
                observation[i, 8] = observation[i, 8] / 0.73
        
        
        
    
        for env, done in zip(self.envs, dones): # Scan for terminated envs
            if done is True:
                terminated_environments.append(env)

        return terminated_environments, rews, infos, observation, dones

    def _send_zero_vel_and_update(self, envs, time_step_update=True):
        """
            send zero velocities to all UNITY envs and update the agent envs

            Note: this function is needed to be called when the manipulator resets to different initial joints
                  positions than the vertical default position. It corrects the dart chain of the envs due to UNITY synchronization

            :param envs: envs
            :param time_step_update: whether to update the timestep of the agents

            :return: obs_converted
        """
        # Send zero velocities to the UNITY envs  #
        if(self.config['env_key'] == 'iiwa_joint_vel'):
            action_dim = self.config['num_joints'] + 1
        elif(self.config['robotic_tool'].find("gripper") == -1):
            action_dim = 7
        else:
            action_dim = 8

        actions = np.zeros((len(envs), action_dim))
        request = self._create_request("ACTION", envs, actions)
        observations = self._send_request(request)
        
        images = self.get_image_from_simulator_obs(observations)
             
        observation = self.return_image_based_observation(images)
        
        
        # Update the agents envs #
        state, _, _, _ = self._update_envs(observations, time_step_update=time_step_update)

        if self.use_mlp:

            for i, env in enumerate(envs):
                mlp_joint_obs = observation[i, :7]
                observation[i, :7] = env.normalize_joints(mlp_joint_obs)
                observation[i, 7] = observation[i, 7] / 1.4
                observation[i, 8] = observation[i, 8] / 0.73



        return observation
    
    
    def get_image_from_simulator_obs(self, observation):

        img_stack = []

        for obs, env in zip(observation, self.envs):

            obs = ast.literal_eval(obs)
            base64_bytes = obs['ImageData'][0].encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            image = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = np.array(image)

            img_stack.append(image)

        return img_stack

    def _get_image_from_simulator_obs(self, observation):

        obs = ast.literal_eval(observation)
        base64_bytes = obs['ImageData'][0].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = np.array(image)

        return image
    
    
    def return_image_based_observation(self,env_images):
        
        # convert cv2 image to numpy array
        all_env_images_array = np.array(env_images)
        # create an empty batch of size same as images from all env, this will be used in line 329
        r3m_input_batch = torch.zeros([self.nenvs,3,224,224], dtype = torch.float64)
        
        # convert the numy images of all envs to PIL Image as torchvision.transforms takes PIL image as input
        for i in range(self.nenvs):
            PIL_Img = Image.fromarray(all_env_images_array[i])
            # width, heigt = PIL_Img.size
            # print(f"width{width}, height{heigt}")
            Img = self.apply_transform_to_images_r3m(PIL_Img)
            # print(Img.shape)
            # add the transformed images to the empty batch created above
            r3m_input_batch[i, : , :, :] = Img
        # feed the batch to r3m and get the latent vector and feed latent vector to mlp to get the 9 state observation

        # latent_vector_tensor = self.r3m(r3m_input_batch)
        # mlp_output = self.mlp_model(latent_vector_tensor)

        r3m_input_batch = r3m_input_batch.to(self.device)
        r3m_input_batch = r3m_input_batch.float()
        observation = self.model(r3m_input_batch)

        observation = observation.detach().cpu().numpy()
        
        return observation

    def _return_image_based_observation(self, env_images):

        # convert cv2 image to numpy array
        all_env_images_array = np.array(env_images)
        # create an empty batch of size same as images from all env, this will be used in line 329
        r3m_input_batch = torch.zeros([1, 3, 224, 224], dtype=torch.float64)

        # convert the numy images of all envs to PIL Image as torchvision.transforms takes PIL image as input
        for i in range(1):
            PIL_Img = Image.fromarray(all_env_images_array[i])
            # width, heigt = PIL_Img.size
            # print(f"width{width}, height{heigt}")
            Img = self.apply_transform_to_images_r3m(PIL_Img)

            # add the transformed images to the empty batch created above
            r3m_input_batch[i, :, :, :] = Img
        # feed the batch to r3m and get the latent vector and feed latent vector to mlp to get the 9 state observation

        # latent_vector_tensor = self.r3m(r3m_input_batch)
        # mlp_output = self.mlp_model(latent_vector_tensor)

        r3m_input_batch = r3m_input_batch.to(self.device)
        r3m_input_batch = r3m_input_batch.float()
        observation = self.model(r3m_input_batch)

        observation = observation.detach().cpu().numpy()

        return observation



    def _update_envs(self, observations, time_step_update):
        """
            update the agent envs (dart chain, time_step etc.) given the new UNITY observations
 
            :param observations: UNITY format
            :param time_step_update: whether to update the timestep of the agents

            :return: terminated_environments, rews, infos, obs_converted, dones 
        """
        env_stack = []
        # acc_tensor = torch.zeros(32,9)
        # i = 0
        for obs, env in zip(observations, self.envs):
            obs = ast.literal_eval(obs)
            # obs['mlp_observation'] = mlp_obs
            env_stack.append(env.update(obs, time_step_update))

        return [list(param) for param in zip(*env_stack)]

    ###############
    # End helpers #
    ###############

    def reset_task(self):
        pass

    def close(self):
        SimulatorVecEnv._client.terminate()

    def __len__(self):
        return self.nenvs

    def render(self, mode=None):
        """
            Override default vectorized render behaviour. SB3 renders all envs into a single window. This behaviour is
            incompatible with our task_monitor.py implementation. Instead render each env into separate windows
        """
        if(self.config["env_key"] != 'iiwa_joint_vel'):
            for env in self.envs:
                if(env.dart_sim.enable_viewer): # Render is active
                    env.render()

    # Calling destructor
    # def __del__(self):
    #     print("Destructor called")
    #     self.env_process.terminate()
