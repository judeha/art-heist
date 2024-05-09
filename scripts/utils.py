#%%

import torch
import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Dict, Set, Union, Optional, Callable

#%%
def generate_schedule(n_heists: int, schedule_size: int) -> Any:
    """ Builds a schedule for thief

    Parameters
    ---------------------------
    n_heists        : int
        number of heists to generate
    schedule_size   : int
        scheduling window size

    Output
    ----------------------------
    schedule        : np.array (bool)
        0 = available, 1 = busy
    """
    
    # init empty schedule
    schedule = np.zeros((schedule_size,))
    if not n_heists:
        return schedule
    
    intvl = schedule_size // n_heists # prevents squashing all heists at end
    max_idx = intvl
    min_idx = 0
    for _ in range(n_heists):
        start_idx = np.random.randint(min_idx, max_idx)
        end_idx = np.random.randint(start_idx, max_idx)
        schedule[start_idx: end_idx] = 1

        # update intervals
        max_idx += intvl
        min_idx + end_idx
    return schedule

#%%
def generate_qual(qual_min: int, qual_max: int, qual_size: int) -> Any:
    """ Builds qualification vector for thief or slot data

    Parameters
    ---------------------------
    qual_min  : int
        minimum qualification possible
    qual_max  : int
        maximum qualification possible
    qual_size : int
        size of qualification vector
    
    Output
    ---------------------------
    qual      : np.array (float)
        qual[i] = x implies qualification level possessed/required in role i is x
    """
    # NOTE: can replace with other generation function

    # Generate each qualification independently
    return np.random.randint(qual_min, qual_max, qual_size)

#%%
def generate_sat_mask(sat_size: int) -> Any:
    """ Builds job satisfaction mask for thief or slot_data
    
    Output
    ---------------------------
    sat_mask: np.array (bool)
        0 = not a factor in job satisfaction, 1 = is a factor
    """
    return np.random.randint(0,2,sat_size).astype('float')

#%%
def generate_sat(sat_min: int, sat_max: int, sat_size: int) -> Any:
    """ Build job satisfaction vector for thief data
    
    Output
    ---------------------------
    sat: np.array (float) of size (4,)
        sat[i] = x implies satisfaction in category i is level x
    """
    sat = generate_sat_mask(sat_size)
    sat *= np.random.uniform(sat_min, sat_max, sat_size)
    return sat

#%%
def generate_heist_data(schedule_size: int,
                        heist_dur_max: int,
                        n_slots_max  : int,
                        n_slots_min  : int,
                        start_time   : Union[int,None]=None,
                        end_time     : Union[int,None]=None,
                        n_slots      : Union[int,None]=None,
                        n_slots_req  : Union[int,None]=None) -> Any:
    """ Generates vector for single heist

    Parameters: optional for custom data generation
    ---------------------------
    schedule_size: int
        scheduling window size
    heist_dur_max: int
        maximum time a heist can last
    n_slots_max  : int
        maximum slots allowed on heist
    n_slots_min  : int
        minimum slots allowed on heist
    start_time   : int
    end_time     : int
    n_slots      : int
        number of slots left on heist
    n_slots_req  : int
        number of required slots left

    Output
    ---------------------------
    np.array (float) of size (4,)
    """

    if start_time  is None: start_time = np.random.randint(0, schedule_size)
    if end_time    is None: end_time = np.random.randint(start_time, start_time + heist_dur_max)
    if n_slots     is None: n_slots = np.random.randint(n_slots_min, n_slots_max)
    if n_slots_req is None: n_slots_req = np.random.randint(0, n_slots) 

    return np.array([start_time, end_time, n_slots, n_slots_req])
#%%
def generate_slot_data(qual_max : int,
                       qual_min : int,
                       qual_size: int,
                       sat_size : int,
                       required : int=0,
                       quals    : Union[Any,None]=None,
                       sat_mask : Union[Any,None]=None) -> Any:
    """ Generates vector for single slot

    Parameters
    ---------------------------
    required : int
        0 = slot is not required, 1 = required
    quals    : np array or None
        optional for custom qualification requirements
    sat_mask : np array or None
        optional for custom job satisfaction mask 

    Output
    ---------------------------
    np.array (float) of size (1 + qual_size + sat_size,)
    """
    required = np.expand_dims(np.array(required), axis=0)
    if quals is None: quals = generate_qual(qual_min, qual_max, qual_size)
    if sat_mask is None: sat_mask = generate_sat_mask(sat_size)
    return np.concatenate([required, quals, sat_mask])

#%%
def generate_thief_data(schedule_size: int,
                        n_heists_max: int,
                        n_heists_min: int,
                        qual_max : int,
                        qual_min : int,
                        qual_size: int,
                        sat_max  : int,
                        sat_min  : int,
                        sat_size : int,
                        schedule : Union[Any,None]=None,
                        quals    : Union[Any,None]=None,
                        sat      : Union[Any,None]=None) -> Any:
    """ Generates vector for single slot

    Parameters
    ---------------------------
    n_heists_max: int
        max number of heists a thief can be assigned to
    n_heists_min: int
        min number of heists a thief can be assigned to
    schedule: np array or None
        optional for custom schedule
    quals   : np array or None
        optional for custom qualifications
    sat     : np array or None
        optional for custom job satisfaction 

    Output
    ---------------------------
    np.array (float) of size (schedule_size + qual_size + sat_size,)
    """
    if schedule is None:
        n_heists = np.random.randint(n_heists_min, n_heists_max)
        schedule = generate_schedule(n_heists, schedule_size)
    if quals is None: quals = generate_qual(qual_min, qual_max, qual_size)
    if sat   is None: sat = generate_sat(sat_min, sat_max, sat_size)
    return np.concatenate([schedule, quals, sat])
# %%
def conflict_schedule(schedule: Any,
                      s: int,
                      e: int) -> bool:
    
    """ Checks if interval conflicts with schedule """
    # True if conflict
    return schedule[s:e].sum() > 0

def conflict_interval(s1, e1, s2, e2) -> bool:
    
    """ Checks if intervals conflict """
    no_conflict = (s1 < s2 and e1 < s2) or (s2 < s1 and e2 < s1)
    # True if conflict
    return not no_conflict
#%%
def is_unqualified(thief_quals: Any,
                   slot_quals : Any) -> bool:
    """ Checks if thief is unqualified for slot """
    # True if unqualified
    return (thief_quals < slot_quals).any()
# %%
def remove_edges(edge_index: torch.tensor, remove_idx: Any) -> Any:
    """ Given edge_index tensor, array of idx in ascending order to remove -> return new view of edge_index """
    keep_idx = np.arange(edge_index.shape[1])
    keep_idx = np.delete(keep_idx, remove_idx)
    return edge_index[:,keep_idx]