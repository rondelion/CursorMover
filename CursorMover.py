# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Header.CursorMover
import os

import gymnasium
import numpy as np
import sys
import argparse
import json
import brica1.brica_gym
import brical

from collections import deque

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import CBT1cCA_2


class StreamDataSet(torch.utils.data.IterableDataset):
    def __getitem__(self, index):
        pass

    def __init__(self, stream):
        super(StreamDataSet).__init__()
        self.stream = stream

    def __iter__(self):
        return iter(self.stream)


# Object recognizer:Compress visual input into latent variables
class ObjectRecognizer(brica1.brica_gym.Component):
    def __init__(self, fovea_size, config):
        super().__init__()
        self.make_in_port('observation', 36300)
        self.make_in_port('token_in', 1)
        self.make_out_port('object_lv', 20)
        self.make_out_port('token_out', 1)
        use_cuda = not config["no_cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.fovea_size = fovea_size
        fovea_shape = (-1, 3, fovea_size, fovea_size)
        self.model_type = config['type']
        if config['type'] == "sparse":
            from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder
            self.model = SparseAutoencoder(fovea_shape, config['model_config']).to(self.device)
        elif config['type'] == "simple":
            from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder
            self.model = SimpleAutoencoder(fovea_shape, config['model_config']).to(self.device)
        elif config['type'] == "neu_beta":
            from neu_VED import VariationalEncoderDecoder
            config["model_config"]["input_dim"] = 3 * fovea_size * fovea_size
            config["model_config"]["output_dim"] = 3 * fovea_size * fovea_size
            config["model_config"]["device"] = self.device
            ae = VariationalEncoderDecoder(config["model_config"])
            self.model = ae.model
        else:
            raise NotImplementedError('Fovea model not supported: ' + str(config['type']))
        if os.path.isfile(config["model_file"]):
            self.model.load_state_dict(torch.load(config["model_file"]))
        else:
            raise FileNotFoundError('Fovea model file not found: ' + config["model_file"])
        self.model.eval()
        self.prev_observation = None    # for debug

    def fire(self):
        img = self.get_in_port('observation').buffer
        # get the fovea image
        center = img.shape[0] // 2
        half_fovea_size = self.fovea_size // 2
        fovea = img[center - half_fovea_size:center + half_fovea_size,
                    center - half_fovea_size:center + half_fovea_size]
        fovea = fovea.transpose((2, 0, 1))
        fovea = fovea.reshape(1, *fovea.shape)
        # get the latent variables
        data = torch.from_numpy(fovea.astype(np.float64)).float().to(self.device)
        if self.model_type == "neu_beta":
            data = torch.flatten(data)
            _, object_lv, _ = self.model(data, None)
        else:
            object_lv, _ = self.model(data)
        self.results['object_lv'] = object_lv.to('cpu').detach().numpy().ravel()
        self.prev_observation = img   # for debug

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']


# Fovea diff predictor:Predict the saliency map
class FoveaDiffPredictor(brica1.brica_gym.Component):
    def __init__(self, device, config, writer):
        super().__init__()
        self.make_in_port('cursor_action', 10)
        self.make_in_port('gaze_shift', 9)
        self.make_in_port('object_lv', 10)
        self.make_in_port('token_in', 1)
        self.make_out_port('prediction_error', 1)
        self.make_out_port('token_out', 1)
        self.model_file = config["model_file"] if "model_file" in config else None
        self.learn_mode = config["learn_mode"]
        self.in_data = None
        self.device = device
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.log_interval = config['log_interval']
        self.writer = writer
        self.stream = deque()
        self.cnt = 0
        self.loss = 0.0  # for logging
        from SimplePredictor import Perceptron
        self.model = Perceptron(config['input_dim'],
                                config['hidden_dim'],
                                1)
        if self.model_file is not None and os.path.isfile(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
        import torch.nn as nn
        self.loss_func = nn.BCELoss()
        from torch import optim
        self.model.optimizer = optim.Adam(self.model.parameters())
        self.threshold = config['threshold']
        self.prev_object_lv = None
        self.scene_size = config['scene_size']
        self.max_jump = self.scene_size // 2
        self.prev_gaze_shift = np.zeros(2, dtype=int)

    def fire(self):
        gaze_shift = self.get_in_port('gaze_shift').buffer
        cursor_action = self.get_in_port('cursor_action').buffer
        cursor_action = torch.from_numpy(cursor_action.astype(np.float32)).clone()
        object_lv = torch.from_numpy(self.get_in_port('object_lv').buffer.astype(np.float32)).clone()
        token = self.get_in_port('token_in').buffer[0]
        if token > 2 and sum(np.abs(self.prev_gaze_shift)) == 0:   # token > 2 and no saccade
            diff = torch.abs(torch.sum(object_lv - self.prev_object_lv)) / len(object_lv)
            diff = 1.0 if diff >= self.threshold else 0.0
            diff = torch.from_numpy(np.array([diff]).astype(np.float32)).clone()
            if self.learn_mode:
                self.learn(self.in_data, diff)
            predicted = self.model(self.in_data)
            self.loss = self.loss_func(predicted, diff).item()
            self.results['prediction_error'] = np.array([self.loss])
        else:
            self.results['prediction_error'] = np.array([0], dtype=float)
        self.in_data = torch.cat((object_lv, cursor_action))
        self.prev_object_lv = object_lv
        self.prev_gaze_shift = gaze_shift

    def learn(self, in_data, diff):
        in_out_torch = (in_data.clone().detach().float(),
                        diff.clone().detach().float())
        self.stream.append(in_out_torch)
        average_loss: float = 0.0
        if self.cnt % self.batch_size == 0 and self.cnt != 0:
            dataset = StreamDataSet(self.stream)
            loss_sum = 0.0
            cnt = 0
            for epoque in range(self.epochs):
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch_idx, (x, y) in enumerate(train_loader):
                    self.model.train()
                    self.model.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    self.model.optimizer.step()
                    loss_sum += loss.item()
                    cnt += 1
            average_loss = loss_sum / (cnt * torch.numel(y))
            self.stream.clear()
        if self.cnt % self.log_interval == 0 and self.cnt != 0:
            print('Cnt: {}\tLoss: {:.6f}'.format(self.cnt, average_loss))
            self.writer.add_scalar('SaliencyMapPredictor/loss', average_loss, self.cnt)
        self.cnt += 1

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']

    def close(self):
        if self.model_file is not None and self.learn_mode:
            torch.save(self.model.state_dict(), self.model_file)


# Surprise reward:Compute reward as change x surprise
class SurpriseReward(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.object_lv_size = config['object_lv_size']
        self.make_in_port('object_lv', self.object_lv_size)
        self.make_in_port('prediction_error', 1)
        self.make_in_port('token_in', 1)
        self.make_out_port('reward', 1)
        self.prev_object_lv = None
        self.threshold = config['threshold']

    def fire(self):
        object_lv = self.get_in_port('object_lv').buffer
        prediction_error = self.get_in_port('prediction_error').buffer[0]
        if self.prev_object_lv is not None:
            diff = abs((object_lv - self.prev_object_lv).sum()) / len(object_lv)
            diff = 1.0 if diff >= self.threshold else 0.0
            self.results['reward'] = np.array([diff * prediction_error])
        else:
            self.results['reward'] = np.array([0], dtype=float)
        self.prev_object_lv = object_lv

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.inputs['prediction_error'] = np.array([0.0])
        self.inputs['object_lv'] = np.zeros(self.object_lv_size, dtype=float)
        self.results['token_out'] = np.array([0])
        self.results['reward'] = np.array([0], dtype=float)
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_in_port('object_lv').buffer = self.inputs['object_lv']
        self.get_in_port('prediction_error').buffer = self.inputs['prediction_error']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.get_out_port('reward').buffer = self.results['reward']
        self.prev_object_lv = None


# Action output:Integrate gaze and cursor action
class ActionOutput(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.make_in_port('token_in', 1)
        self.make_in_port('cursor_action', 10)
        self.make_in_port('gaze_shift', 25)
        self.make_out_port('action', 35)
        self.make_out_port('token_out', 1)

    def fire(self):
        cursor_action1h = self.get_in_port('cursor_action').buffer  # one-hot vector
        cursor_action = np.zeros(3, dtype=int)
        tmp = np.zeros(5, dtype=int)
        if np.amax(cursor_action1h) > 0:
            if np.amax(cursor_action1h[4:]) > 0:    # grab = 1
                tmp = cursor_action1h[4:]
                cursor_action[2] = 1
            elif np.amax(cursor_action1h[:4]) > 0:  # grab = 0
                tmp = cursor_action1h[:4]
            pos = np.argmax(tmp)
            cursor_action[0] = (pos // 2) * 2 - 1
            cursor_action[1] = (pos % 2) * 2 - 1
        gaze_shift = self.get_in_port('gaze_shift').buffer
        action = np.concatenate([gaze_shift, cursor_action])
        self.results['action'] = action

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']


def main():
    parser = argparse.ArgumentParser(description='An agent that acts with a cursor')
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=50, metavar='N',
                        help='Max steps in an episode (default: 30)')
    parser.add_argument('--config', type=str, default='CursorMover.json', metavar='N',
                        help='Configuration (default: CursorMover.json')
    parser.add_argument('--brical', type=str, default='CursorMover.brical.json', metavar='N',
                        help='a BriCAL json file')
    parser.add_argument('--use-cuda', default=False,
                        help='uses CUDA for training')
    parser.add_argument('--dump_flags', type=str, default="",
                        help='f:fovea, c:cursor action, b:bg, o:obs, p:predictor')
    parser.add_argument('--no_render', action='store_true',
                        help='Env image rendering')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    writer = SummaryWriter()  # for TensorBoard
    train = {"episode_count": args.episode_count, "max_steps": None, "writer": writer,
             "dump_flags": args.dump_flags}
    if args.dump is not None and args.dump_flags != "":
        try:
            dump = open(args.dump, mode='w')
        except IOError:
            print('Error: No dump path specified', file=sys.stderr)
            sys.exit(1)
    else:
        dump = None
    train["dump"] = dump
    config['train'] = train
    if "e" in args.dump_flags:
        config['env']['dump'] = dump
    else:
        config['env']['dump'] = None
    if "p" in args.dump_flags:
        config['agent']['CursorActor']['NeoCortex']['ActionPredictor']['dump'] = dump
    else:
        config['agent']['CursorActor']['NeoCortex']['ActionPredictor']['dump'] = None

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config["device"] = device
    config['agent']['CursorActor']['device'] = device

    stage_size = config["env"]["stage_size"]
    scene_size = stage_size * 2 - 1
    # saliency_map_shape = (scene_size, scene_size)
    # max_jump = stage_size - 1
    if config['agent']['ObjectRecognizer']['type'] == "neu_beta":
        object_lv_size = config['agent']['ObjectRecognizer']['model_config']['z_dim']
    else:
        object_lv_size = config['agent']['ObjectRecognizer']['model_config']['num_units']
    cursor_n_action = config['agent']['CursorActor']['n_action']
    config['agent']['FoveaDiffPredictor']['input_dim'] = object_lv_size + cursor_n_action
    config['agent']['FoveaDiffPredictor']['scene_size'] = scene_size
    config['agent']['SurpriseReward']['object_lv_size'] = object_lv_size
    config['agent']['CursorActor']['in_dim'] = object_lv_size

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!")
        exit(-1)

    env = gymnasium.make(config['env']['name'], config=config['env'], render_mode="human")

    nb.unit_dic['CursorMover.Periphery2Saliency'].__init__(config)
    nb.unit_dic['CursorMover.PriorityMap2Gaze'].__init__(config)
    nb.unit_dic['CursorMover.ObjectRecognizer'].__init__(config['env']['grid_size'],
                                                         config['agent']['ObjectRecognizer'])
    nb.unit_dic['CursorMover.FoveaDiffPredictor'].__init__(config["device"], config['agent']['FoveaDiffPredictor'],
                                                           writer)
    nb.unit_dic['CursorMover.SurpriseReward'].__init__(config['agent']['SurpriseReward'])
    nb.unit_dic['CursorMover.CursorActor'].__init__(config['agent']['CursorActor']['learning_mode'], train,
                                                    config['agent']['CursorActor'])
    nb.unit_dic['CursorMover.ActionOutput'].__init__(config)
    nb.make_ports()

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['CursorMover.CognitiveArchitecture']
    model.make_in_port('reward', 1)
    model.make_in_port('done', 1)
    agent = agent_builder.create_gym_agent(nb, model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    for i in range(train["episode_count"]):
        last_token = 0
        print("Episode: ", i)
        prev_action = None
        for j in range(args.max_steps):
            scheduler.step()
            current_token = agent.get_out_port('token_out').buffer[0]
            # print(i, j, current_token, agent.env.done)
            if last_token + 1 == current_token:
                last_token = current_token
                if not args.no_render and current_token > 1 and not agent.env.done:
                    # print("render")
                    env.render()
                if dump is not None and "f" in args.dump_flags and current_token > 2:
                    dump.write("{0}\tLoss: {1:.2f}\tObs.Diff: {2:.2f}\tLast Action: {3}\n".
                               format(i,
                                      nb.unit_dic['CursorMover.FoveaDiffPredictor'].loss,
                                      nb.unit_dic['CursorMover.Periphery2Saliency'].diff,
                                      prev_action))
                if dump is not None and "c" in args.dump_flags and current_token > 2:
                    if sum(np.abs(prev_action[:2])) == 0 and sum(np.abs(prev_action[2:])) > 0:
                        env_obj = env.env.env.env
                        dump.write("{}\tActor Loss: {:.2f}\tObs.Diff: {:.2f}\tReward: {:.2f}\tLast Action: {}\
\t{}\t{}\t{}\t{}".
                                   format(i,
                                          nb.unit_dic['CursorMover.CursorActor'].neoCortex.action_predictor.average_loss,
                                          nb.unit_dic['CursorMover.Periphery2Saliency'].diff,
                                          nb.unit_dic['CursorMover.SurpriseReward'].results['reward'][0],
                                          prev_action,
                                          env_obj.gaze,
                                          env_obj.pos_xy,
                                          env_obj.cursor_xy,
                                          env_obj.suits[:env_obj.cardinality]
                                          ).replace('\n', ''))
                        dump.write("\n")
                prev_action = agent.get_out_port('action').buffer
            if agent.env.done:
                break
        agent.env.flush = True
        nb.unit_dic['CursorMover.Periphery2Saliency'].reset()
        nb.unit_dic['CursorMover.PriorityMap2Gaze'].reset()
        nb.unit_dic['CursorMover.ObjectRecognizer'].reset()
        nb.unit_dic['CursorMover.FoveaDiffPredictor'].reset()
        nb.unit_dic['CursorMover.SurpriseReward'].reset()
        nb.unit_dic['CursorMover.CursorActor'].reset()
        nb.unit_dic['CursorMover.ActionOutput'].reset()
        # TODO: WRITE END OF EPISODE CODE (component reset etc.) HERE!!
        agent.env.reset()
        agent.env.done = False

    # nb.unit_dic['CursorMover.Periphery2Saliency'].close()
    # nb.unit_dic['CursorMover.PriorityMap2Gaze'].close()
    # nb.unit_dic['CursorMover.ObjectRecognizer'].close()
    nb.unit_dic['CursorMover.FoveaDiffPredictor'].close()
    # nb.unit_dic['CursorMover.SaliencySurpriseReward'].close()
    nb.unit_dic['CursorMover.CursorActor'].close()
    # nb.unit_dic['CursorMover.ActionOutput'].close()
    print("Close")
    env.close()


if __name__ == '__main__':
    main()
