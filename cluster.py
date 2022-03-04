import pandas as pd
import numpy as np

from jenkspy import JenksNaturalBreaks

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

pd.options.mode.chained_assignment = "raise"

class __CompetitionDataProcesser:
    def prep_data(self, data):
        dt = data.copy(deep = True)
        dt.loc[:, "task_no"] = dt["task_date"].astype("category").cat.codes
        dt.loc[:, "solution_date"] = pd.to_datetime(dt["solution_date"])
        dt = dt[dt["task_no"] != 0]
        dt = dt[dt["solution_is_last"] == 1]
        dt = dt.sort_values(by="solution_date")
        return dt

class __ClusteringModel(__CompetitionDataProcesser):
    def __init__(self):
        self.predicted = False

    def fit_predict(self, data):
        self.fit(self, data)
        return self.predict(self)

    def fit(self, data):
        self.dt = self.prep_data(data)
        self.n_tasks = len(pd.unique(self.dt["task_no"]))

    def predict(self):
        self.predicted = True
        return [self.group_in_task(n) for n in range(1, self.n_tasks+1)]

    def compute_time_delta(self, data):
        time_delta = data["solution_date"] - data["solution_date"][0]
        return time_delta.dt.total_seconds()

    def get_task_data(self, n):
        task_data = self.dt.groupby("task_no").get_group(n)
        task_data = task_data.reset_index()
        return task_data.groupby("solution_answer")

    def group_in_task(self, n):
        raise NotImplementedError()

    def get_keys(self, task_data):
        raise NotImplementedError()

    def group_in_key(self, key_data):
        raise NotImplementedError()

class JNBClustering(__ClusteringModel):
    def get_keys(self, task_data):
        keys = task_data.groups.keys()
        flt = lambda k: True if task_data.get_group(k).shape[0] > 6 else False
        jenk_ready, jenk_small = [], []
        for k in keys:
            jenk_ready.append(k) if flt(k) else jenk_small.append(k)
        return [jenk_ready, jenk_small]
    
    def group_in_key(self, key_data):
        time_delta = self.compute_time_delta(key_data)

        jnb = JenksNaturalBreaks()
        jnb.fit(time_delta)
        breaks = jnb.group(time_delta)

        grouping = np.cumsum([len(b) for b in breaks])

        return np.split(key_data["participant_id"], grouping)

    def group_in_task(self, n):
        task_data = self.get_task_data(n)
        keys = self.get_keys(task_data)

        task_groups = []

        for k in keys[0]:
            key_data = task_data.get_group(k).reset_index()
            task_groups.append(self.group_in_key(key_data))

        for k in keys[1]:
            gr = task_data.get_group(k)
            task_groups.append(np.split(gr["participant_id"], 1))

        return task_groups

class KDEClustering(__ClusteringModel):
    def __init__(self, sensitivity=1, kde_params=False):
        super().__init__()
        self.kde_params = kde_params

        if sensitivity < 0 or sensitivity > 1:
            raise ValueError("Sensitivity parameter must be between 0 and 1.")
        self.sensitivity = sensitivity

    def __get_kde(self):
        if not self.kde_params:
            return KernelDensity()
        else:
            return KernelDensity(self.kde_params)

    def group_in_task(self, n):
        task_data = self.get_task_data(n)
        keys = self.get_keys(task_data)

        task_groups = []

        for k in keys:
            key_data = task_data.get_group(k).reset_index()
            task_groups.append(self.group_in_key(key_data))

        return task_groups

    def group_in_key(self, key_data):
        time_delta = self.compute_time_delta(key_data)
        data = np.array(time_delta).reshape(-1,1)

        kde = self.__get_kde().fit(data)
        num = int(np.max(24*60*self.sensitivity))
        est = kde.score_samples(np.linspace(0, 24*60*60, num=num).reshape(-1,1))
        minima = argrelextrema(est, np.less)[0]
        breaks = np.histogram(time_delta, [-np.infty, *minima, np.infty])[0]

        grouping = np.cumsum(breaks[breaks != 0])

        return np.split(key_data["participant_id"], grouping)

    def get_keys(self, task_data):
        return task_data.groups.keys()

class ClusteringAnalysis(__CompetitionDataProcesser):
    def __init__(self, clustering, data):
        self.clustering = clustering
        self.data = self.prep_data(data)
        self.n_tasks = len(pd.unique(self.data["task_no"]))

    def __pair_sim_score(self, participants):
        parts = set(participants)
        
        score = 0
        for task in self.clustering:
            for ans in task:
                for grp in ans:
                    if parts.issubset(set(grp)):
                        score = score + 1

        return (score, score/self.n_tasks)

    def compute_similarity_score(self, participants):
        if len(participants) == 2:
            return self.__pair_sim_score(participants)
        else:
            d = len(participants)
            shape = [d,d]
            score_matrix = np.empty(shape)
            for p in participants:
                for q in participants:
                    sc = self.__pair_sim_score([p,q])
                    p_idx = participants.index(p)
                    q_idx = participants.index(q)
                    score_matrix[p_idx, q_idx] = sc[1]
            return score_matrix
