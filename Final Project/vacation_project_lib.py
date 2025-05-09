#!/usr/bin/env python3.8
#
# Author: Erin Walker
#
#########################
# 
import numpy as np 

def price(a,b,c,d,e,f,g, max_money):
    # List to store all Valid Combinations 
    combinations = []
    spare_change_values = []

    # List to Store Failed Combinations 
    failed_combinations = []
    failed_costs = []
    for park_count in range(2): # will only list the activity zero or one time.
        for pool_count in range(2):
            for castle_count in range(2):
                for cave_count in range(2):
                    for museum_count in range(2):
                        for city_count in range(2):
                            for garden_count in range(2):

                                # Calculate the Total Cost for This Combination
                                total_cost = (
                                    park_count * a +
                                    pool_count * b +
                                    castle_count * c + 
                                    cave_count * d + 
                                    museum_count * e + 
                                    city_count * f +
                                    garden_count * g
                                )

                                # Check if the total cost is within the budget
                                if total_cost <= max_money:
                                    spare_change = max_money - total_cost
                                    combinations.append(
                                        (park_count, pool_count, castle_count, cave_count, museum_count, city_count, garden_count)
                                    )
                                    spare_change_values.append(spare_change)
                                else:
                                    failed_combinations.append(
                                    (park_count, pool_count, castle_count, cave_count, museum_count, city_count, garden_count)
                                    )
                                    failed_costs.append(total_cost)

    spare_change_values = np.array(spare_change_values)
    combinations = np.array(combinations)
    least_spare_change_index = spare_change_values.argmin() # Where the minimum occurs
    best_combo = combinations[least_spare_change_index]
                                
    return spare_change_values, combinations, least_spare_change_index, best_combo, failed_combinations, failed_costs
                                





def time(a,b,c,d,e, max_time):
    # List to Store all Valid Combinations
    combinations = []
    spare_time_values = []

    # List to Store Failed Combinations 
    failed_combinations = []
    failed_times = []
    # Possible outcomes using the best combo answer found above. 
    for park_count in range(2): # will only list the activity zero or one time.
            for castle_count in range(2):
                for cave_count in range(2):
                    for museum_count in range(2):
                        for garden_count in range(2):

                            # Calculate the Total Cost for This Combination
                            total_time = (
                                park_count * a +
                                castle_count * b +
                                cave_count * c +
                                museum_count * d +
                                garden_count * e
                            )

                            # Check if the total cost is within the budget
                            if total_time <= max_time:
                                spare_time = max_time - total_time
                                combinations.append(
                                    (park_count, castle_count, cave_count, museum_count, garden_count)
                                )
                                spare_time_values.append(spare_time)
                            else:
                                failed_combinations.append(
                                    (park_count, castle_count, cave_count, museum_count, garden_count)
                                )
                                failed_times.append(total_time)
    spare_time_values = np.array(spare_time_values)
    combinations = np.array(combinations)
    least_spare_time_index = spare_time_values.argmin()
    best_time_combo = combinations[least_spare_time_index]

    return spare_time_values, combinations, least_spare_time_index, best_time_combo, failed_combinations, failed_times

# def something that will call on the time answer then input it into 
# the price answer then the time answer again ??

