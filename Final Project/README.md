# Budgeting and Timing a Vacation

This repository optimizes a resource with given restraints. In this case, the resource is the budget and the restraints are prices and time. 

## Table of Contents 

1. [Description](#description)
2. [Installation](#installation)
3. [Useage](#useage)
   - [Vacation Library](#vacation-library)
   - [Vacation Code](#vacation-code)

## Description 

This project is made to help someone on a budget going on vacation. With a budget set aside to do activities, one can max their experience by doing as many activities as possible. After setting a budget, and seeing what activities would be best, those activities are then put into a similar loop but for time restraints instead of money. This part tells you what activities are within your vacation time set aside for activities. This code is not to help you save money put to maximize its use. 

## Installation 

For this code to work, you need to download the vacation_project_lib.py and import it into the Final_Project.py file. You also need to import numpy. To make sure the code works properly, you need to first run the best price combinations before doing anything to do with time. Once the price loop has run, you can then add the time code underneath it with the outcome of best combo. 

## Useage

### Vacation Library 

In the vacation_project_lib.py, a price and time function are given. These need to be changed to your own activities, prices, and times. If you are adding activities, or sometime just changing the ones given, an error may appear that says you forgot a comma. You did not. To fix this you will need to rewrite the variable it is having trouble with. In the library file this will happen at total cost.\
The varaibles in the price function are the prices of each activity and the max money you are willing to spend. Similarly, the time function variables are the times, in hours, of each activity and the max time you can spend doing activities. The for loop makes sure each activity can only be done once or not at all. If you want to see if you can do an activity more than once, you will need to change the range from 2 to one more than the number of times your want it to repeat.

### Price Loop

'''python 
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
'''

### Vacation Code 

Once your library is set up, it needs to be imported to the .py file. Then, make a list of the prices and times of each activity. State what your max money and time is, then make it a float. If desirable, the time can be changed from hours to minutes. Next, call on the library you made to find what activities fit in your budget best and filling in the variables. To find the best combo of activities, the activity count in best combo needs to be multiplied by the price of the activity. Do this for each activity and add them together. 




