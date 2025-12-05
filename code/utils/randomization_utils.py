## functions modified from https://stackoverflow.com/questions/93353/create-many-constrained-random-permutation-of-a-list
import math, random
import numpy as np

def get_pool(items, n_elements_per_subject, use_each_times):
    pool = {}
    for n in items:
        pool[n] = use_each_times
    
    return pool

def rebalance(ret, pool, n_elements_per_subject):
    max_item = None
    max_times = None
    
    for item, times in pool.items():
        if max_times is None:
            max_item = item
            max_times = times
        elif times > max_times:
            max_item = item
            max_times = times
    
    next_item, times = max_item, max_times

    candidates = []
    for i in range(len(ret)):
        item = ret[i]

        if next_item not in item:
            candidates.append( (item, i) )
    
    swap, swap_index = random.choice(candidates)

    swapi = []
    for i in range(len(swap)):
        if swap[i] not in pool:
            swapi.append( (swap[i], i) )
    
    which, i = random.choice(swapi)
    
    pool[next_item] -= 1
    pool[swap[i]] = 1
    swap[i] = next_item

    ret[swap_index] = swap

def create_balanced_orders(items, n_elements_per_subject, use_each_times, consecutive_limit=2,  error=1):
    '''
    Returns a set of unique lists under the constraints of 
    - n_elements_per_subject (must be less than items)
    - use_each_times: number of times each item should be seen across subjects

    Together these define the number of subjects

    '''

    n_subjects = math.ceil((use_each_times * len(items)) / n_elements_per_subject)

    print (f'Creating orders for {n_subjects} subjects')

    pool = get_pool(items, n_elements_per_subject, use_each_times)
    
    ret = []
    while len(pool.keys()) > 0:
        while len(pool.keys()) < n_elements_per_subject:
            rebalance(ret, pool, n_elements_per_subject)
        
        selections = sorted(random.sample(pool.keys(), n_elements_per_subject))
        
        for i in selections:
            pool[i] -= 1
            if pool[i] == 0:
                del pool[i]

        ret.append( selections )
        
        unique, counts = np.unique(ret, return_counts=True)
        
        if all(np.logical_and(counts <= use_each_times + error, counts >= use_each_times)):
               break
    return ret

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_consecutive_list_idxs(orders, consecutive_length):
    
    # Find lists with consecutive items violating our constraint
    idxs = np.where([np.any(np.asarray(list(map(len, consecutive(order)))) >= consecutive_length) for order in orders])[0]
    
    return idxs

def sort_consecutive_constraint(orders, consecutive_length=3):
    
    # Get sets of all orders
    all_order_idxs = np.arange(len(orders))
    
    # Find lists with consecutive items violating our constraint
    consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_length)
    
    while len(consecutive_order_idxs):

        for order_idx in consecutive_order_idxs:
            # Select the current list violating the constraint
            current_list = np.asarray(orders[order_idx])

            random_list_options = np.setdiff1d(all_order_idxs, order_idx)

            # Find all sets of consecutive items in the current list --> find their lengths
            consecutive_items = consecutive(current_list)
            consecutive_lengths = np.asarray(list(map(len, consecutive_items)))

            # Find sets of slices that violate the constraint
            violations = np.where(consecutive_lengths >= consecutive_length)[0]

            for violation in violations:
                # Select items that need to be swapped --> these will be swapped into a randomly selected list
                swap_items = consecutive_items[violation][1::2]

                for item in swap_items:
                    swap_idx = np.where(current_list == item)[0]

                    # Select a random other list
                    random_list_idx = random.choice(random_list_options)
                    random_list = np.asarray(orders[random_list_idx])

                    # Find choices not within our current list
                    swap_choices = np.setdiff1d(random_list, current_list)

                    # Select a random choice
                    choice = random.choice(swap_choices)
                    
                    # Make sure we didn't violate our constraint again with either list
                    while (
                        np.isin(choice,current_list) or 
                        np.isin(item,random_list)
                    ):
                        
                        # Select a random other list
                        random_list_idx = random.choice(random_list_options)
                        random_list = np.asarray(orders[random_list_idx])

                        # Find choices not within our current list
                        swap_choices = np.setdiff1d(random_list, current_list)

                        # Select a random choice
                        choice = random.choice(swap_choices)
                    
                    # Find the index to swap to
                    choice_idx = np.where(random_list == choice)[0]

                    # Swap the two items
                    current_list[swap_idx] = choice
                    random_list[choice_idx] = item

                    # Set them in the overall orders
                    orders[order_idx] = sorted(current_list)
                    orders[random_list_idx] = sorted(random_list)
                    
        # Find lists with consecutive items violating our constraint
        consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_length)
    return orders