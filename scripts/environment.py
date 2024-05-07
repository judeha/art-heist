import pandas as pd
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from functools import partial
from typing import Any, Set, Tuple, List, Dict, Union, Optional
import gc
gc.enable()

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric import EdgeIndex
from torch_geometric.typing import OptPairTensor

%load_ext autoreload
%autoreload 1

import sys
sys.path.append('../scripts')
from utils import generate_slot_data, generate_heist_data, generate_thief_data, conflict_interval, conflict_schedule, is_unqualified, remove_edges

# Vectorize conflict_interval
vconflict_interval = np.vectorize(conflict_interval)

config_problemdef_PATH = '../configs/v1_problemdef.yaml'
with open(config_problemdef_PATH, 'r') as f:
    config_problemdef = yaml.load(f, Loader=yaml.SafeLoader)

# Load generation parameters
config_params_PATH = '../configs/v1_params.yaml'
with open(config_params_PATH, 'r') as f:
    config_params = yaml.load(f, Loader=yaml.SafeLoader)

class CustomEnvironment():
    def __init__(self, config=config_problemdef):
        # Define immutable parameters: node sizes
        self.schedule_size = config['schedule_size']
        self.qual_size = len(config['qualifications'])
        self.sat_size = len(config['job_satisfaction'])

        # Hard code features to idx mappings
        self.featidx_h_start = 0
        self.featidx_h_end = 1
        self.featidx_h_n_slots = 2
        self.featidx_h_n_slots_req = 3
        self.featidx_t_schedule = (0,self.schedule_size)
        self.featidx_t_qual = (self.schedule_size, self.schedule_size + self.qual_size)
        self.featidx_t_sat = (self.schedule_size + self.qual_size, self.thief_size)
        self.featidx_s_req = 0
        self.featidx_s_qual = (1, self.qual_size+1)
        self.featidx_s_sat = (self.qual_size+1, self.slot_size)
            
    def set_params(self, config=config_params):
        # Define mutable parameters for data generation
        heist_dur_max = config['heist_dur_max']
        n_slots_max = config['n_slots_max']
        n_slots_min = config['n_slots_min']
        n_heists_max = config['n_heists_max']
        n_heists_min = config['n_heists_min']
        qual_max = config['qual_max']
        qual_min = config['qual_min']
        sat_max = config['sat_max']
        sat_min = config['sat_min']

        # Set partial functions using problem definition sizes
        self.generate_heist_data = partial(generate_heist_data, self.schedule_size, heist_dur_max, n_slots_max, n_slots_min)
        self.generate_thief_data = partial(generate_thief_data, self.schedule_size, n_heists_max, n_heists_min, qual_max, qual_min, self.qual_size, sat_max, sat_min, self.sat_size)
        self.generate_slot_data = partial(generate_slot_data, qual_max, qual_min, self.qual_size, self.sat_size)

    def reset(self):
        n_thieves = 30
        n_heists = 50

        self.init_indices()
        self.init_dictionaries()
        self.heist_df = self.gen_n_heists(n_heists) # will update index_d_heist1, index_d_heist2 in place
        self.slot_df = self.gen_n_slots() # will update index_c_slot, index_c_heist, heist2slot, slot2heist in place
        self.thief_df = self.gen_n_thieves(n_thieves) # will update index_a_thief, index_a_slot, slot2thief in place

        self.data = self.reset_data()
        self.reward = None
        self.termination = False
        self.timestep = 0

    def step(self, edge_idx): # NOTE: editing self.data inplace except for self.index_b_thief, self.index_b_slot, self.thiefslot2idx
        
        self.reward = 0
        self.timestep += 1

        # Check for termination
        if not self.thiefslot2idx or self.timestep > 999:
            self.termination = True

            # Penalize unsatisfied thieves
            self.reward -= self.data['thief'].x[:,self.featidx_t_sat[0]: self.featidx_t_sat[1]].sum()
            # Penalize grounded assignments
            for h, crew in self.heist2assigned.items():
                  reward_num_assignments = len(crew)
                  reward_time = self.data['heist'].x[h][self.featidx_h_end] - self.data['heist'].x[h][self.featidx_h_start]
                  self.reward -= reward_num_assignments + reward_time
            return self.data, self.reward, self.termination, self.timestep
        
        thief = self.data['thief','possible','slot'].edge_index[0, edge_idx].item() # NOTE: cannot use self.index_a_thief, since it's not updated
        slot = self.data['thief','possible','slot'].edge_index[1, edge_idx].item()
        
        # Step 1: Add (thief, slot) to index_b
        self.index_b_thief.append(thief)
        self.index_b_slot.append(slot)
        self.data['thief','assigned','slot'].edge_index = torch.stack([torch.tensor(self.index_b_thief), torch.tensor(self.index_b_slot)])
        # NOTE: more efficient way?

        # Step 2: Add (thief, slot) to heist2assigned dictionary
        heist = self.slot2heist[slot]
        if self.heist2assigned.get(heist):
            self.heist2assigned[heist].append((thief, slot))
        else: self.heist2assigned[heist] = [(thief, slot)]

        # Step 3: Remove (t, s) edges from index_a based on constraints
        remove_idx = self.constraint2(thief, heist) + self.constraint3(slot)
        self.data['thief','possible','slot'].edge_index = remove_edges(self.data['thief','possible','slot'].edge_index, remove_idx)   
        # Update thiefslot2idx wholesale
        self.thiefslot2idx = {(k1.item(), k2.item()):v for k1,k2,v in zip(self.data['thief','possible','slot'].edge_index[0],self.data['thief','possible','slot'].edge_index[1],np.arange(self.data['thief','possible','slot'].edge_index.shape[1]))}

        # Step 4: Update thief schedule (Already accomplished through Step 2, Substep 3)
        self.data['thief'].x[thief][self.featidx_t_schedule[0]: self.featidx_t_schedule[1]] = 1

        # Step 5: Check for takeoff and calculate reward
        if not self.data['heist'].x[heist][self.featidx_h_n_slots_req]: # already taken off
            self.reward += self.calculate_reward_qual_sat(thief, slot)
            # Update thief information
            self.update_thief_sat(thief, slot)
        elif self.data['heist'].x[heist][self.featidx_h_n_slots_req] == 1 and self.data['slot'].x[slot][self.featidx_s_req]: # currently taking off
            self.reward += self.calculate_reward_qual_sat(thief, slot)
            # Update thief information
            self.update_thief_sat(thief, slot)
            # Update previous crew information
            crew = self.heist2assigned[heist]
            for (t,s) in crew:
                self.reward += self.calculate_reward_qual_sat(t,s)
                # Update previous thieves
                self.update_thief_sat(t,s)
        else: # currently grounded
            self.reward += self.calculate_reward_qual_sat(thief, slot)

        # Step 6: Update heist information
        self.data['heist'].x[heist][self.featidx_h_n_slots] -= 1
        if self.data['slot'].x[slot][self.featidx_s_req]:
            self.data['heist'].x[heist][self.featidx_h_n_slots_req] -= 1

        # print(len(self.thiefslot2idx), self.data['thief','possible','slot'].edge_index.shape)
        return self.data, self.reward, self.termination, self.timestep

    def init_indices(self):
        self.index_a_thief, self.index_a_slot = [], []      # NOTE: DISREGARD AFTER INIT
        self.index_b_thief, self.index_b_slot = [], []
        self.index_c_slot, self.index_c_heist = [], []
        self.index_d_heist1, self.index_d_heist2 = [], []
    
    def init_dictionaries(self):
        self.heist2slot = {}
        self.heist2heist = {}
        self.slot2heist = {}
        self.slot2thief = {}

        # Assignment dictionaries
        self.heist2assigned = {}
        self.thiefslot2idx = {}

    def gen_n_heists(self, n_heists): # TODO: add optional config file of size n_heists x dims
        heist_df = pd.DataFrame()
        for h in range(n_heists):
            # Get heist data
            heist_data = self.generate_heist_data()

            # Add to heist_df
            tmp_df = pd.DataFrame(heist_data).T
            tmp_df.index = [h]
            heist_df = pd.concat([heist_df, tmp_df])

        heist_df.index.rename('heistId', inplace=True)

        # Update index_d and heist2heist dictionary (O(n^2))
        for h, row in heist_df.iterrows():
            conflicts = vconflict_interval(heist_df[self.featidx_h_start].values, heist_df[self.featidx_h_end].values,
                                            row[self.featidx_h_start], row[self.featidx_h_end])                        
            # [conflict_interval(row[featidx_h_start], row[featidx_h_end], x, y) for x,y in zip(heist_df[featidx_h_start].values, heist_df[featidx_h_end].values)]
            indices = np.where(conflicts)[0] # list(np.compress(heist_df.index, conflicts))
            self.heist2heist[h] = indices
            for h2 in indices:
                self.index_d_heist1.append(h)
                self.index_d_heist2.append(h2)

        return heist_df
    
    def gen_n_slots(self): 
        i = 0 # counter of total slots so far
        slot_df = pd.DataFrame()
        # For each heist
        for h, heist in self.heist_df.iterrows():
            # Number of slots per heist
            n_slots: int = int(heist[self.featidx_h_n_slots])
            n_slots_req: int= heist[self.featidx_h_n_slots_req]
            
            # For each slots
            for s in range(n_slots):
                # Get slot data based on required status
                slot_data: Any = self.generate_slot_data(required=(s < n_slots_req))

                # Update index_c
                self.index_c_slot.append(i)
                self.index_c_heist.append(h)

                # Update heist2slot and slot2heist dictionaries
                if h in self.heist2slot: self.heist2slot[h].append(i)
                else: self.heist2slot[h] = [i]
                self.slot2heist[i] = h

                # Add to slot_df
                tmp_df = pd.DataFrame(slot_data).T
                tmp_df.index = [i]
                slot_df = pd.concat([slot_df, tmp_df])

                i += 1 # update slot counter
            
        slot_df.index.rename('slotId', inplace=True)
        return slot_df
    
    def gen_n_thieves(self, n_thieves): 
        i = 0 # counter of total edges
        thief_df = pd.DataFrame()
        # For each thief
        for t in range(n_thieves):
            # Get thief data
            thief_data = self.generate_thief_data()
            schedule = thief_data[self.featidx_t_schedule[0]: self.featidx_t_schedule[1]]

            # Create thief-slot edges
            # For each heist
            for h, heist in self.heist_df.iterrows():
                # If schedule conflict: no edge
                if conflict_schedule(schedule, heist[self.featidx_h_start], heist[self.featidx_h_end]):
                    continue
                # If no schedule conflict: get indices of slot on heist
                slot_idx = self.heist2slot[h]
                # For each eligible slot
                for s in slot_idx:
                    # If thief unqualified: no edge
                    thief_qual = thief_data[self.featidx_t_qual[0]: self.featidx_t_qual[1]]
                    slot_qual = self.slot_df.iloc[s, self.featidx_s_qual[0]: self.featidx_s_qual[1]]
                    if is_unqualified(thief_qual, slot_qual):
                        continue
                    # If thief qualified: update index_a
                    self.index_a_thief.append(t)
                    self.index_a_slot.append(s)
                    # Update slot2thief dictionary
                    if self.slot2thief.get(s): self.slot2thief[s].append(t) 
                    else: self.slot2thief[s] = [t]                    
                    # Update thiefslot2idx dictionary
                    self.thiefslot2idx[(t,s)] = i
                    i += 1
        
            # Add to thief_df
            tmp_df = pd.DataFrame(thief_data).T
            tmp_df.index = [t]
            thief_df = pd.concat([thief_df, tmp_df])

        thief_df.index.rename('thiefId', inplace=True)
        return thief_df

    def reset_data(self):
        index_a = torch.stack([torch.tensor(self.index_a_thief), torch.tensor(self.index_a_slot)])
        index_b = torch.stack([torch.tensor(self.index_b_thief), torch.tensor(self.index_b_slot)])
        index_c = torch.stack([torch.tensor(self.index_c_slot), torch.tensor(self.index_c_heist)])
        index_d = torch.stack([torch.tensor(self.index_d_heist1), torch.tensor(self.index_d_heist2)])

        data = HeteroData()

        # Add node indices
        data['thief'].node_id = torch.tensor(self.thief_df.index)
        data['slot'].node_id = torch.tensor(self.slot_df.index)
        data['heist'].node_id = torch.tensor(self.heist_df.index)

        # Add node features
        data["thief"].x = torch.tensor(self.thief_df.values).to(torch.float)
        data["slot"].x = torch.tensor(self.slot_df.values).to(torch.float)
        data["heist"].x = torch.tensor(self.heist_df.values).to(torch.float)

        # Add edge indices
        data["thief","possible","slot"].edge_index = index_a # has shape (2, num_edges)
        data["thief","assigned","slot"].edge_index = index_b # has shape (2, num_edges)
        data["slot","on","heist"].edge_index = index_c # has shape (2, num_edges)
        data["heist1","conflicts","heist2"].edge_index = index_d # has shape (2, num_edges)

        # Add reverse edge
        data = T.ToUndirected()(data)
        return data

    # def constraint1(self, thief: int, heist: int) -> List:
    #     """ Given (thief, heist) pair, find edges that violate constraint 1: one person per heist """
    #     slots: List = self.heist2slot[slot] if self.heist2slot.get(heist) is not None else []
    #     idx  : List = [self.thiefslot2idx[(thief,s)] for s in slots if (thief,s) in self.thiefslot2idx]
    #     return idx
    
    def constraint2(self, thief: int, heist: int) -> List:
        """ Given (thief, heist) pair, find edges that violate constraint 2: thief cannot be assigned to conflicting heists """
        conflicting_heists: List = self.heist2heist[heist] if heist in self.heist2heist else []
        # print(heist, conflicting_heists)
        slots : List = []
        for h in conflicting_heists:
            slots += (self.heist2slot[h] if h in self.heist2slot else [])
        idx : List = [self.thiefslot2idx[(thief,s)] for s in slots if s if (thief,s) in self.thiefslot2idx]
        return idx

    def constraint3(self, slot: int) -> List:
        """ Given a slot, find edges that violate constraint 3: only one person per slot """
        thieves : List = self.slot2thief.pop(slot) if slot in self.slot2thief else []
        idx : List = [self.thiefslot2idx[(t,slot)] for t in thieves if (t,slot) in self.thiefslot2idx]
        return idx
    
    def calculate_reward_qual_sat(self, thief: int, slot: int) -> float:
        """ Given index for thief and slot, calculate overqual and satisfaction reward """
        thief_satisfaction = self.data['thief'].x[thief][self.featidx_t_sat[0]: self.featidx_t_sat[1]]
        thief_quals        = self.data['thief'].x[thief][self.featidx_t_qual[0]: self.featidx_t_qual[1]]
        slot_satisfaction  = self.data['slot'].x[slot][self.featidx_s_sat[0]: self.featidx_s_sat[1]]
        slot_quals         = self.data['slot'].x[slot][self.featidx_s_qual[0]: self.featidx_s_qual[1]]
        reward_overquals = (thief_quals - slot_quals).sum()
        reward_satisfaction = (thief_satisfaction * slot_satisfaction).sum()
        return reward_satisfaction - reward_overquals

    def update_thief_sat(self, thief: int, slot: int) -> None:
        """ Given index for thief, slot, update thief information in place """
        slot_sat = self.data['slot'].x[slot][self.featidx_s_sat[0]: self.featidx_s_sat[1]]
        self.data['thief'].x[thief][self.featidx_t_sat[0]: self.featidx_t_sat[1]] *= (1 - slot_sat)