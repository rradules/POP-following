import numpy as np
import random


class Stav:
    def __init__(self, state, probab, pcs, action=None, vector=None):
        self.state = state
        self.prob = probab
        self.action = action
        self.vector = vector
        self.tvector = self.prob * vector if vector is not None else None
        self.pcs_table = (pcs.loc[pcs['State'] == state])  # hacky, should be an argument
        self.pcs_list = self.precompute_tuples()
        self.pprune_tuples()
        self.pick_random()

    def __str__(self):
        return f'({self.state}, {self.prob}, {self.action}, {self.action}, {self.vector}, {self.tvector})'

    def pick_random(self):
        row = self.pcs_table.sample().to_numpy()[0]
        self.action = int(row[1])
        self.vector = row[2:]
        self.tvector = self.prob * self.vector

    def pprune_tuples(self):
        new_tuples = []
        while len(self.pcs_list) > 0:
            nn_tups = []
            ctup = self.pcs_list[0]
            for tup in self.pcs_list:
                if np.greater_equal(tup[1], ctup[1]).all():
                    ctup = tup
            for tup in self.pcs_list:
                if not np.greater_equal(ctup[1], tup[1]).all():
                    nn_tups.append(tup)
            new_tuples.append(ctup)
            self.pcs_list = nn_tups
        self.pcs_list = new_tuples

    def precompute_tuples(self):
        result = []
        for index, row in self.pcs_table.iterrows():
            line = row.to_numpy()
            act = int(line[1])
            vec = line[2:]
            tvec = self.prob * vec
            result.append((act, vec, tvec))
        return result

    def improve_step(self, n_vector, cur_vector, cur_score=None):
        cur_score = np.linalg.norm(n_vector - cur_vector) if cur_score is None else cur_score
        min_vec = n_vector - cur_vector + self.tvector
        nw_cur_vector = cur_vector - self.tvector
        for tup in self.pcs_list:
            n_score = np.linalg.norm(min_vec + tup[2])
            if n_score < cur_score:
                self.action = tup[0]
                self.vector = tup[1]
                self.tvector = tup[2]
                cur_score = n_score
                return True, (nw_cur_vector + self.tvector), cur_score
        return False, cur_vector, cur_score


def toStavs(trans, pcs):
    result = []
    for i in range(len(trans)):
        if trans[i] > 0:
            tup = Stav(i, trans[i], pcs)
            result.append(tup)
    return result


def popf_local_search(problem, n_vector, state, cur_vector=None, score=None):
    if cur_vector is None or score is None:
        for stav in problem:
            cur_vector = stav.tvector if cur_vector is None else cur_vector + stav.tvector
        score = np.linalg.norm(n_vector - cur_vector)
    improved = True
    while (improved):
        improved = False
        random.shuffle(problem)
        for stav in problem:
            improved, cur_vector, score = stav.improve_step(n_vector, cur_vector, score)
            if improved:
                break
    for stav in problem:
        if stav.state == state:
            action = stav.action
            value_vector = stav.vector
    return cur_vector, score, action, value_vector


def popf_iter_local_search(problem, n_vector, state, reps=10, pertrub_p=1.0):
    cur_vector = None
    for stav in problem:
        stav.pick_random()
        cur_vector = stav.tvector if cur_vector is None else cur_vector + stav.tvector
        if stav.state == state:
            action = stav.action
            value_vector = stav.vector
    score = np.linalg.norm(n_vector - cur_vector)
    for i in range(reps):
        for stav in problem:
            rnumber = random.random()
            if rnumber < pertrub_p:
                stav.pick_random()
        cur_vector_rep, score_rep, action_rep, value_vector_rep = popf_local_search(problem, n_vector, state)
        if score_rep < score:
            cur_vector = cur_vector_rep
            score = score_rep
            action = action_rep
            value_vector = value_vector_rep
    return cur_vector, score, action, value_vector
