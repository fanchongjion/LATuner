import json
from lhs import LHSGenerator
import subprocess
import time
import mysql.connector
import os
import re
import numpy as np
from shutil import copyfile
from logger import SingletonLogger
import queue
import pandas as pd
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qKnowledgeGradient
from botorch.optim import optimize_acqf
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import LatinHypercubeInitialDesign
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from pathlib import Path
from openai import OpenAI
from mab import ThompsonSamplingBandit
import requests

def transform_knobs2vector(knobs_detail, knobs):
    keys = list(knobs.keys())
    ys = []
    for key in keys:
        if knobs_detail[key]['type'] == 'integer':
            minv, maxv = knobs_detail[key]['min'], knobs_detail[key]['max']
            tmpv = (knobs[key] - minv) / (maxv - minv)
            ys.append(tmpv)
        elif knobs_detail[key]['type'] == 'enum':
            enum_vs = knobs_detail[key]['enum_values']
            tmpv = enum_vs.index(knobs[key]) / (len(enum_vs) - 1)
            ys.append(tmpv)
        else:
            pass
    return ys
def transform_vector2knobs(knobs_detail, vector):
    keys = list(knobs_detail.keys())
    knobs = {}
    for i in range(len(keys)):
        if knobs_detail[keys[i]]['type'] == 'integer':
            minv, maxv = knobs_detail[keys[i]]['min'], knobs_detail[keys[i]]['max']
            tmpv = (maxv - minv) * float(vector[i]) + minv
            knobs[keys[i]] = int(tmpv)
        elif knobs_detail[keys[i]]['type'] == 'enum':
            enum_vs = knobs_detail[keys[i]]['enum_values']
            tmpv = vector[i] * (len(enum_vs) - 1)
            knobs[keys[i]] = enum_vs[int(tmpv)]
        else:
            pass
    return knobs
def transform_knobs2cnf(knobs_detail, knobs):
    keys = list(knobs.keys())
    for key in keys:
        if knobs_detail[key]['type'] == 'integer':
            knobs[key] = int(knobs[key])
        else:
            pass
    return knobs

def proxy_chat(system_content, prompt):
    url = "https://api.openai-hk.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": os.getenv("openai_key")
    }
    data = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
    result = response.content.decode("utf-8")
    return json.loads(result)

class Tuner():
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        self.knobs_config_path = knobs_config_path
        self.knob_nums = knob_nums
        self.knob_idxs = knob_idxs
        self.initialize_knobs()
        self.dbenv = dbenv
        self.bugets = bugets
        self.logger = None if not self.dbenv else self.dbenv.logger
    def initialize_knobs(self):
        f = open(self.knobs_config_path)
        knob_tmp = json.load(f)
        KNOB_DETAILS = {}
        if not self.knob_idxs:
            i = 0
            while i < self.knob_nums:
                key = list(knob_tmp.keys())[i]
                KNOB_DETAILS[key] = knob_tmp[key]
                i = i + 1
        else:
            if type(self.knob_idxs[0]) == int:
                for idx in self.knob_idxs:
                    key = list(knob_tmp.keys())[idx]
                    KNOB_DETAILS[key] = knob_tmp[key]
            else:
                for key in self.knob_idxs:
                    KNOB_DETAILS[key] = knob_tmp[key]
        f.close()
        self.knobs_detail = KNOB_DETAILS

class LHSTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "LHS"
    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_gen = LHSGenerator(lhs_num, self.knobs_detail)
        lhs_configs = lhs_gen.generate_results()
        return lhs_configs
    def tune(self):
        self.dbenv.step(None)
        knobs_set = self.lhs(self.bugets)
        for knobs in knobs_set:
            self.dbenv.step(knobs)
            
class GridTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "Grid"

    def _grid_search(self, params_list, results, current_params=None):
        if current_params is None:
            current_params = []
        if not params_list:
            return current_params
        current_dimension = params_list[0]
        for value in current_dimension:
            result = self._grid_search(params_list[1:], results, current_params + [value])
            if result:
                results.append(result)
    
    def sampling(self, interval):
        knobs_list = []
        for knob_name in self.knobs_detail.keys():
            type = self.knobs_detail[knob_name]["type"]
            if type == "integer":
                minv = self.knobs_detail[knob_name]["min"]
                maxv = self.knobs_detail[knob_name]["max"]
                knobs_list.append(list(np.linspace(minv, maxv, interval, dtype=np.int32)))
            else:
                knobs_list.append(self.knobs_detail[knob_name]["enum_values"])
        results = []
        self._grid_search(knobs_list, results)
        return results
    
    def tune(self, interval=10):
        self.dbenv.step(None)
        knobs_set = self.sampling(interval)
        keys = list(self.knobs_detail.keys())
        for rd, ss in enumerate(knobs_set):
            self.logger.info(f"tuning round {rd + 1} begin!!")
            knobs = {}
            for i in range(len(keys)):
                if isinstance(ss[i], np.integer):
                    knobs[keys[i]] = int(ss[i])
                else:
                    knobs[keys[i]] = ss[i]
            self.dbenv.step(knobs)
            self.logger.info(f"tuning round {rd + 1} over!!")

class GPTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=20):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "GP"
        self.objective = objective
        self.warm_start_times = warm_start_times
    
    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_gen = LHSGenerator(lhs_num, self.knobs_detail)
        lhs_configs = lhs_gen.generate_results()
        return lhs_configs

    def _get_next_point(self):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        #EI = ExpectedImprovement(gp, best_f=train_Y.max())
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        knobs = transform_vector2knobs(self.knobs_detail, candidate[0])
        
        return knobs
    def tune(self):
        self.dbenv.step(None)
        knobs_set = self.lhs(self.warm_start_times)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs = self._get_next_point()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

class SMACTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=20):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "SMAC"
        self.objective = objective
        self.warm_start_times = warm_start_times
    
    def _train(self, config, seed=0):
        knobs = dict(config)
        knobs = transform_knobs2cnf(self.knobs_detail, knobs)
        metric = self.dbenv.step(knobs)
        if self.objective == 'lat':
            return metric
        else:
            return -metric
    def tune(self):
        self.dbenv.step(None)
        keys = list(self.knobs_detail.keys())
        knobs = []
        for key in keys:
            if self.knobs_detail[key]["type"] == "integer":
                knobs.append(Float(key, (self.knobs_detail[key]["min"], self.knobs_detail[key]["max"]), default=self.knobs_detail[key]["default"]))
            elif self.knobs_detail[key]["type"] == "enum":
                knobs.append(Categorical(key, self.knobs_detail[key]["enum_values"], default=self.knobs_detail[key]["default"]))
            else:
                pass
        configspace = ConfigurationSpace("smac_tuning", seed=0)
        configspace.add_hyperparameters(knobs)
        scenario = Scenario(configspace, n_trials=self.bugets, output_directory=Path(self.dbenv.results_save_dir))
        smac = HyperparameterOptimizationFacade(scenario, self._train, initial_design=LatinHypercubeInitialDesign(scenario, self.warm_start_times))
        incumbent = smac.optimize()
        return incumbent
    
class LLMTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=10, prune_nums=5):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "LLM"
        self.objective = objective
        self.warm_start_times = warm_start_times
        self.prune_nums = prune_nums
        self.proxy = True
        api_key = os.getenv("openai_key")
        self.client = proxy_chat if self.proxy else OpenAI(api_key=api_key)
        self.system_content = '''You will be helping me with the knob tuning task for {0} database. '''
        self.user_content_prune = '''The specific information for the knobs is: {0}. The specific information for the machine on which the {1} works is: {2} cores {3} RAM and {4} disk. The specific information for the workload is: {5} size {6}. The goal of the current tuning task is to optimize {7}, please give the {8} knobs that have the greatest impact on the performance of the database. You should give these top-{8} knobs by json style.  The given knobs must be included in the previously given knobs. Just give the json without any other extra output.'''
        self.user_content_ws_samples = '''The specific information for the knobs is: {0}. The specific information for the machine on which the {1} works is: {2} cores {3} RAM and {4} disk. The specific information for the workload is: {5} size {6}. The goal of the current tuning task is to optimize {7}, please suggest {8} diverse yet effective configurations to initiate a Bayesian Optimization process for knobs tuning. You mustn't include “None” in the configurations. Your response should include a list of dictionaries, where each dictionary describes one recommended configuration.Just give the dictionaries without any other extra output.'''
        self.bandit = ThompsonSamplingBandit(num_arms=2)

    def _check_knobs_valid(self, knobs_set):
        if len(knobs_set) != 5:
            return False 
        keys = list(self.knobs_detail.keys())
        for knobs in knobs_set:
            tmp_keys = list(knobs.keys())
            for key in tmp_keys:
                if key not in keys:
                    self.logger.info(f"key: {key} not in knob space")
                    return False
            for key in tmp_keys:
                if self.knobs_detail[key]['type'] == 'integer':
                    v = knobs[key]
                    minv, maxv = self.knobs_detail[key]['min'], self.knobs_detail[key]['max']
                    if v < minv or v > maxv:
                        self.logger.info(f"knobs range error, {key} : {v}")
                        return False
                else:
                    v = knobs[key]
                    vs = self.knobs_detail[key]['enum_values']
                    if v not in vs:
                        self.logger.info(f"knobs range error, {key} : {v}")
                        return False
        return True
    
    def _gen_candidates_llm(self, nums, target):
        system_content = self.system_content.format("MySQL")
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        prompt = '''The following examples demonstrate {0} database running on a machine with {1} cores, {2} of memory, and a {3} disk, under a {4} {5} workload. These examples involve adjusting various knobs configurations to observe changes in {6} metrics:\n'''.format("MySQL", 4, "8GB", "1TB", "2GB", "TPC-C", obj_str)
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            prompt += "Knob configuration: " + json.dumps(knobs) + "\n"
            prompt += "Performance: " + str(int(metric)) + "\n"
        prompt +=  f"The database knob space is: {json.dumps(self.knobs_detail)}." + "\n"
        prompt += f"Please recommend {nums} configurations that will result in a database {obj_str} of {int(target)}. Each knob must contained within the knob space, Your response must only contain the predicted configurations, in the format ## Knob configuration: ##."
        count = 10
        while count > 0:
            try:
                if self.proxy:
                    completion = self.client(system_content, prompt)
                else:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt}
                        ]
                    )
                strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
                pattern = re.compile(r'\{.*?\}')
                matches = pattern.findall(strs)
                samples = []
                for match in matches:
                    samples.append(eval(match))
                if self._check_knobs_valid(samples):
                    self.logger.info(f"time {10 - count}, gpt return sucess")
                    break
            except Exception as e:
                print(e)
                pass
            time.sleep(15)
            self.logger.info(f"time {10 - count}, gpt return error")
            count -= 1

        if count == 0:
            self.logger.error(f"gpt return fail")
            return
        return samples

    def _prediction_llm(self, knobs_set):
        system_content = self.system_content.format("MySQL")
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        prompt = '''The following examples demonstrate {0} database running on a machine with {1} cores, {2} of memory, and a {3} disk, under a {4} {5} workload. These examples involve adjusting various knobs configurations to observe changes in {6} metrics:\n'''.format("MySQL", 4, "8GB", "1TB", "2GB", "TPC-C", obj_str)
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            prompt += "Knob configuration: " + json.dumps(knobs) + "\n"
            prompt += "Performance: " + str(int(metric)) + "\n"
        prompt +=  f"The allowable ranges for knobs are: {json.dumps(self.knobs_detail)}. "
        prompt += "Please combine the above information to determine which of the following configurations is a high potential configuration: \n"
        for knobs in knobs_set:
            prompt += json.dumps(knobs) + "\n"
        prompt += "Your response should only contain one of the above configurations."
        count = 10
        while True:
            try:
                if self.proxy:
                    completion = self.client(system_content, prompt)
                else:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt}
                        ]
                    )
                strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
                pattern = re.compile(r'\{.*?\}')
                matches = pattern.findall(strs)
                samples = []
                for match in matches:
                    samples.append(eval(match))
                if len(samples) > 0:
                    break
            except Exception as e:
                print(e)
            count -= 1

        return samples[0]

    def _knob_prune(self, nums):
        knobs_str = json.dumps(self.knobs_detail)
        system_content = self.system_content.format("MySQL")
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        user_content = self.user_content_prune.format(knobs_str, "MySQL", '4', '8GB', '1TB', '2GB', 'TPC-C', obj_str, nums)
        if self.proxy:
            completion = self.client(system_content, user_content)
        else:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
        strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
        keys = []
        for key in list(self.knobs_detail.keys()):
            if key in strs:
                keys.append(key)
        print(keys)
        assert(len(keys) == nums)
        KNOB_DETAILS = {}
        for key in keys:
            KNOB_DETAILS[key] = self.knobs_detail[key]
        self.knobs_detail = KNOB_DETAILS
        self.knob_nums = nums
    
    def _get_warm_start_samples(self, nums):
        knobs_str = json.dumps(self.knobs_detail)
        system_content = self.system_content.format("MySQL")
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        user_content = self.user_content_ws_samples.format(knobs_str, "MySQL", '4', '8GB', '1TB', '2GB', 'TPC-C', obj_str, nums)
        if self.proxy:
            completion = self.client(system_content, user_content)
        else:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
        strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
        print(strs)
        pattern = re.compile(r'\{.*?\}')
        matches = pattern.findall(strs)
        samples = []
        for match in matches:
            samples.append(eval(match))
        knobs = samples
        return knobs
    
    def _get_next_point_origin(self):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        target = float(train_Y.min()) if self.objective == "lat" else float(train_Y.max())
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        #EI = ExpectedImprovement(gp, best_f=target, maximize=False if self.objective == 'lat' else True)
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=10, raw_samples=2000
            )
        knobs = transform_vector2knobs(self.knobs_detail, candidate[0])
        
        return knobs
    
    def _get_next_point_llm_assist(self, candidate_nums=5):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        target = float(train_Y.min()) if self.objective == "lat" else float(train_Y.max())
        print(target)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        EI = ExpectedImprovement(gp, best_f=target, maximize=False if self.objective == 'lat' else True)
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidates_default, acq_values_default = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=candidate_nums, raw_samples=2000, return_best_only=False
            )
        llm_initial_samples = self._gen_candidates_llm(candidate_nums, target)
        samples = []
        for sample in llm_initial_samples:
            samples.append(transform_knobs2vector(self.knobs_detail, sample))
        if samples:
            samples = torch.tensor(samples, dtype=torch.float64)
            samples = samples.reshape(candidate_nums, 1, self.knob_nums)
            with gpytorch.settings.cholesky_jitter(1e-1):
                candidates_llm, acq_values_llm = optimize_acqf(
                    EI, bounds=bounds, q=1, num_restarts=candidate_nums, batch_initial_conditions=samples, return_best_only=False
                )
            candidates = torch.concat([candidates_default, candidates_llm], dim=0)
            acq_values = torch.concat([acq_values_default, acq_values_llm], dim=0)
        else:
            candidates = candidates_default
            acq_values = acq_values_default

        idx = int(acq_values.argmax())
        knobs_set = []
        size = candidates.shape[0]
        for i in range(size):
            candidate = candidates[i][0]
            tmp_knobs = transform_vector2knobs(self.knobs_detail, candidate)
            knobs_set.append(tmp_knobs)

        knobs = transform_vector2knobs(self.knobs_detail, candidates[idx][0])
        return knobs, knobs_set
    
    def _get_next_point_hybrid(self):
        knobs_default, knobs_set = self._get_next_point_llm_assist()
        knobs_llm = self._prediction_llm(knobs_set)
        return knobs_default, knobs_llm

    def _get_reward(self):
        if self.objective == 'lat':
            perf_first = self.perfs['last_best_lat']
            perf_last = self.perfs['cur_lat']
            if perf_first - perf_last > 0:
                return 1
            else:
                return 0
        else:
            perf_first = self.perfs['last_best_tps']
            perf_last = self.perfs['cur_tps']
            if perf_last - perf_first > 0:
                return 1
            else:
                return 0
        

    def tune(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs = self._get_next_point_origin()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

    def tune_llm_assist(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs, _ = self._get_next_point_llm_assist()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

    def tune_end2end(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        total_reward = 0
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs_default, knobs_llm = self._get_next_point_hybrid()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            chosen_arm = self.bandit.choose_arm()
            if chosen_arm == 0:
                self.logger.info(f"choose arm {chosen_arm}: default knobs!")
                self.dbenv.step(knobs_default)
            else:
                self.logger.info(f"choose arm {chosen_arm}: llm knobs!")
                self.dbenv.step(knobs_llm)

            reward = self._get_reward()
            self.logger.info(f"get reward: {reward}")
            total_reward += reward
            self.bandit.update_arm(chosen_arm, reward)
        self.logger.info(f"total reward: {total_reward}")

class MySQLEnv():
    def __init__(self, host, user, passwd, dbname, workload, objective, method, stress_test_duration, template_cnf_path, real_cnf_path):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.workload = workload
        self.objective = objective
        self.method = method
        self.stress_test_duration = stress_test_duration
        self.template_cnf_path = template_cnf_path
        self.real_cnf_path = real_cnf_path

        self.tolerance_time = 20 #seconds
        self._initial()

    def _initial(self):
        self.timestamp = time.time()    
        self.round = 0
        results_save_dir = f"/home/root3/Tuning/{self.workload}_{self.timestamp}"
        self.results_save_dir = results_save_dir
        if not os.path.exists(results_save_dir):
            os.mkdir(results_save_dir)
        self.metric_save_path = os.path.join(results_save_dir, f'results_{self.objective}.res')
        self.dbenv_log_path = os.path.join(results_save_dir, 'dbenv.log')
        self.stress_results = os.path.join(results_save_dir, 'stress_results')
        self.stress_logs = os.path.join(results_save_dir, 'stress_logs')
        self.tensorboard_logs = os.path.join("/home/root3/Tuning", 'tb_logs')
        if not os.path.exists(self.stress_results):
            os.mkdir(self.stress_results)
        if not os.path.exists(self.stress_logs):
            os.mkdir(self.stress_logs)
        self.logger = SingletonLogger(self.dbenv_log_path).logger
        self.writer = SummaryWriter(log_dir=self.tensorboard_logs, flush_secs=10)
        self.perfs = {}
        self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'], self.perfs["last_best_tps"] = None, None, None, None
        self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'], self.perfs["last_best_lat"] = None, None, None, None

    def _start_mysqld(self):
        proc = subprocess.Popen(['mysqld', '--defaults-file={}'.format(self.real_cnf_path)])
        self.pid = proc.pid
        #print("pid", self.pid)
        count = 0
        start_sucess = True
        self.logger.info('wait for connection')
        time.sleep(1)
        while True:
            try:
                conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
                if conn.is_connected():
                    conn.close()
                    self.logger.info('Connected to MySQL database')
                    self.logger.info('mysql is ready!')
                    self.dbsize = self.get_db_size()
                    self.logger.info(f"{self.workload} database size now is {self.dbsize} MB")
                    break
            except Exception as e:
                print(e)

            time.sleep(1)
            count = count + 1
            self.logger.warn("retry connect to DB")
            if count > 600:
                start_sucess = False
                self.logger.error("can not connect to DB")
                break

        return start_sucess
    
    def _kill_mysqld(self):
        mysqladmin = "/home/root3/mysql/bin/mysqladmin"
        sock = "/home/root3/mysql/mysql.sock"
        cmd = '{} -u{} -S {} shutdown'.format(mysqladmin, self.user, sock)
        p_close = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_close.communicate(timeout=20)
            ret_code = p_close.poll()
            if ret_code == 0:
                self.logger.info("mysqladmin close database successfully")
        except:
            self.logger.warn("force close database by kill -9!!!")
            os.system("ps aux | grep mysqld | grep my.cnf | awk '{print $2}'|xargs kill -9")
        self.logger.info("mysql is shut down")
    
    def get_db_size(self):
        db_conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(self.dbname)
        cmd = db_conn.cursor()
        cmd.execute(sql)
        res = cmd.fetchall()
        db_size = float(res[0][0][:-2])
        db_conn.close()
        return db_size
    
    def replace_mycnf(self, knobs=None):
        if knobs == None:
            copyfile(self.template_cnf_path, self.real_cnf_path)
            return
        f = open(self.template_cnf_path)
        contents = f.readlines()
        f.close()
        for key in knobs.keys():
            contents.append(f"{key}={knobs[key]}")
        strs = '\n'.join(contents)
        with open(self.real_cnf_path, 'w') as f:
            f.write(strs)
            f.flush()
        self.logger.info("replace mysql cnf file")

    def apply_knobs(self, knobs=None):
        self._kill_mysqld()
        self.replace_mycnf(knobs)
        time.sleep(10)
        success = self._start_mysqld()
        return success
    
    def get_workload_info(self):
        with open("./workloads.json", "r") as f:
            infos = json.load(f)
        if self.workload.startswith("benchbase"):
            infos[self.workload]["cmd"] = infos[self.workload]["cmd"].format(time.time(), self.stress_results, self.stress_logs)
            return infos[self.workload]["cmd"]
        else:
            pass
    
    def parser_metrics(self, path):
        if self.workload.startswith("benchbase"):
            with open(path, "r") as f:
                metrics = json.load(f)
        else:
            pass
        return metrics

    def clean_and_find(self):
        files = os.listdir(self.stress_results)
        if self.workload.startswith("benchbase"):
            info_files = [file for file in files if file.endswith("samples.csv")]
            info_file = sorted(info_files)[-1]
            df = pd.read_csv(os.path.join(self.stress_results, info_file))
            self.tps_std = df["Throughput (requests/second)"].std()
            self.lat_std = df["95th Percentile Latency (microseconds)"].std()
            for file in files:
                if not file.endswith("summary.json"):
                    os.remove(os.path.join(self.stress_results, file))

            files = [file for file in files if file.endswith("summary.json")]
            files = sorted(files)
            return os.path.join(self.stress_results, files[-1])
        else:
            pass


    def get_metrics(self):
        cmd = self.get_workload_info()
        self.logger.info(f"get workload stress test cmd: {cmd}")
        self.logger.info("begin workload stress test")
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=self.stress_test_duration + self.tolerance_time)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                self.logger.info("benchmark finished!")
        except Exception as e: 
            self.logger.info(f"{e}")
            return None

        self.logger.info("clean extra files and get metrics file path")
        outfile_path = self.clean_and_find()
        self.logger.info("parser metrics file")
        metrics = self.parser_metrics(outfile_path)
        return metrics

    def step(self, knobs=None):
        self.logger.info(f"round {self.round} begin!!!")
        self.logger.info(f"ready to apply new knobs: {knobs}")
        flag = self.apply_knobs(knobs)
        self.logger.info("apply new knobs success")
        metrics = self.get_metrics()
        if metrics == None:
            self.logger.error("this round stress test fail")
            self.logger.info("round over!!!")
            return
        try:
            if self.workload.startswith("benchbase"):
                metrics["tps_std"] = self.tps_std
                metrics["lat95_std"] = self.lat_std
                metrics['knobs'] = knobs
                metrics['dbsize'] = self.dbsize
                tmp_tps = metrics["Throughput (requests/second)"]
                tmp_lat = metrics["Latency Distribution"]["95th Percentile Latency (microseconds)"]
                if not self.perfs['cur_tps']:
                    self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'], self.perfs['last_best_tps'] = tmp_tps, tmp_tps, tmp_tps, tmp_tps
                else:
                    self.perfs['cur_tps'] = tmp_tps
                    if self.perfs['best_tps'] < tmp_tps:
                        self.perfs['last_best_tps'] = self.perfs['best_tps']
                        self.perfs['best_tps'] = tmp_tps

                if not self.perfs['cur_lat']:
                    self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'], self.perfs['last_best_lat'] = tmp_lat, tmp_lat, tmp_lat, tmp_lat
                else:
                    self.perfs['cur_lat'] = tmp_lat
                    if self.perfs['best_lat'] > tmp_lat:
                        self.perfs['last_best_lat'] = self.perfs['best_lat']
                        self.perfs['best_lat'] = tmp_lat
                    
                self.writer.add_scalars(f"tps_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_tps'], 'best': self.perfs['best_tps'], 'default': self.perfs['default_tps']}, self.round)
                self.writer.add_scalars(f"lat_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_lat'], 'best': self.perfs['best_lat'], 'default': self.perfs['default_lat']}, self.round)
            else:
                pass
        except Exception as e:
            tmp_tps = -0x3f3f3f3f
            tmp_lat = 0x3f3f3f3f
            print(e)
        
        self.save_running_res(metrics)
        self.logger.info(f"save running res to {self.metric_save_path}")
        self.logger.info(f"round {self.round} over!!!")
        self.round += 1

        return tmp_tps if self.objective == "tps" else tmp_lat

    def save_running_res(self, metrics):
        if self.workload.startswith("benchbase"):
            save_info = json.dumps(metrics)
            with open(self.metric_save_path, 'a+') as f:
                f.write(save_info + '\n')
                f.flush()
        else:
            pass

def grid_tuning_task(knobs_idxs=None):
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'grid', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    if not knobs_idxs:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10)
    else:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10, knobs_idxs)
    logger = dbenv.logger
    logger.warn("grid tuning begin!!!")
    grid_tuner.tune()
    logger.warn("grid tuning over!!!")

def lhs_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'lhs', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    lhs_tuner = LHSTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 1000)
    logger = dbenv.logger
    logger.warn("lhs tuning begin!!!")
    lhs_tuner.tune()
    logger.warn("lhs tuning over!!!")

def gp_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'gp', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    lhs_tuner = GPTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 100, None, 'tps', 10)
    logger = dbenv.logger
    logger.warn("gp tuning begin!!!")
    lhs_tuner.tune()
    logger.warn("gp tuning over!!!")

def smac_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'smac', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    smac_tuner = SMACTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 100, None, 'tps', 10)
    logger = dbenv.logger
    logger.warn("smac tuning begin!!!")
    smac_tuner.tune()
    logger.warn("smac tuning over!!!")

def llm_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 60, dbenv, 100, None, 'tps', 10, 5)
    logger = dbenv.logger
    logger.warn("llm tuning begin!!!")
    llm_tuner.tune()
    logger.warn("llm tuning over!!!")

def llm_assist_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm_assist', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 5, dbenv, 37, ['innodb_buffer_pool_size','innodb_write_io_threads','innodb_flush_log_at_timeout','innodb_read_io_threads','innodb_io_capacity_max'], 'tps', 0, 5)
    logger = dbenv.logger
    logger.warn("llm assist tuning begin!!!")
    llm_tuner.tune_llm_assist()
    logger.warn("llm assist tuning over!!!")

def llm_tuning_end2end():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm_end2end', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 60, dbenv, 100, None, 'tps', 10, 5)
    logger = dbenv.logger
    logger.warn("llm end2end tuning begin!!!")
    llm_tuner.tune_end2end()
    logger.warn("llm end2end tuning over!!!")

class TaskQueue():
    def __init__(self, nums=-1):
        self.queue = queue.Queue(nums)

    def _execute_task(self, task):
        task_func, task_args = task
        task_func(*task_args)
    
    def add(self, task):
        self.queue.put(task)
    
    def run(self):
        while not self.queue.empty():
            task = self.queue.get()
            self._execute_task(task)
    

if __name__ == '__main__':
    task_queue = TaskQueue()
    task_queue.add((llm_tuning_task, ()))
    task_queue.run()