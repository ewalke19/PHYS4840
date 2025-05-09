# Budgeting and Timing a Vacation

This repository optimizes a resource with given restraints. In this case, the resource is the budget and the restraints are prices and time. 

## Table of Contents 

1. [Description](#description)
2. [Installation](#installation)
3. [Useage](#useage)
   - [Vacation Library](#vacation-library)
      - [Price Loop](#price-loop)
      - [Time Loop](#time-loop)
   - [Vacation Code](#vacation-code)
      - [Running the Price Loop in the Main Code](#running-the-time-loop-in-the-main-code)
      - [Adding the Time Loop to the Main Code](#adding-the-time-loop-to-the-main-code)
4. [Contributions](#contributions)
5. [Acknowledgments](#acknowledgments)

## Description 

This project is made to help someone on a budget going on vacation. With a budget set aside to do activities, one can maximize their experience by doing as many activities as possible. After setting a budget, and seeing what activities would be best, those activities are then put into a similar loop but for time restraints instead of money. This part tells you what activities are within your vacation time set aside for activities. This code is not to help you save money put to maximize its use. 

## Installation 

For this code to work, you need to download the vacation_project_lib.py and import it into the Final_Project.py file. You also need to import numpy into the library. To make sure the code works properly, you need to first run the best price combinations before doing anything to do with time. Once the price loop has run, you can then add the time code underneath it with the outcome of best combo. 

## Useage

### Vacation Library 

In the vacation_project_lib.py, a price and time function are given. These need to be changed to your own activities, prices, and times. If you are adding activities, or changing the ones given, an error may appear that says you forgot a comma. You did not. To fix this you will need to rewrite the variable it is having trouble with. In the library file this will happen at total_cost. The varaibles in the price function are the prices of each activity and the max money you are willing to spend. Similarly, the time function, the variables are the times, in hours, of each activity and the max time you can spend doing activities. The for loop makes sure each activity can only be done once or not at all. If you want to see if you can do an activity more than once, you will need to change the range from 2 to one more than the number of times your want it to repeat.

#### Price Loop

```python 

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
```

#### Time Loop 

```python

import numpy as np

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

                            # Calculate the Total Time for This Combination
                            total_time = (
                                park_count * a +
                                castle_count * b +
                                cave_count * c +
                                museum_count * d +
                                garden_count * e
                            )

                            # Check if the total time is within the limit
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
```

### Vacation Code 

Once your library is set up, it needs to be imported to the Final_Project.py file. Start by making a list of the prices and times of each activity. State what your max money and time is, then make it a float. If desirable, the time can be changed from hours to minutes. Next, call on the library you made to find what activities fit in your budget best and filling in the variables. To find the best combo of activities, the activity count in best combo needs to be multiplied by the price of the activity. Do this for each activity and add them together. 

#### Running the Price Loop in the Main Code

This first set of code will tell you what activities will max your budget. If there is a 1 printed by the name of the activity, it is a part of the best combo outcome. If a 0 is printed, it is not a part of the combination. This code will also show you what combinations were tried, failed, and what that cost would have been.

```python
# Possible Combos for Each Activity Without Repeating One. Modify as needed for extra activities.  

import vacation_project_lib as vpl 

# List total money set aside for activities
max_money = 250
max_money = float(max_money)

# Example prices and activities. Replace with your own. 
national_park_price = 80
swimming_pool_price = 15
castle_tour_price = 70 
cave_tour_price = 35
museum_tour_price = 40
city_tour_price = 15
garden_tour_price = 25


# List total time set aside for activities 
max_time = 16 # in hours 
max_time = float(max_time)

# Example times with activities. Replace with yoir own data 
national_park_time = 8 # in hours
swimming_pool_time = 2 
castle_tour_time = 5 
cave_tour_time = 3
museum_tour_time = 3
city_tour_time = 2
garden_tour_time = 2

# YOU HAVE TO RUN THE COST CODE FIRST THEN ADD THE TIME CODE

# Imput variables
p = vpl.price(national_park_price, swimming_pool_price, castle_tour_price, cave_tour_price, museum_tour_price, city_tour_price, garden_tour_price, max_money)

#Total Cost of Best Combination 
best_combo_cost = (
    p[3][0] * national_park_price +      # p[3] calls on the best_combo from the function price
    p[3][1] * swimming_pool_price +      # the second [] calls on the activity in best_combo
    p[3][2] * castle_tour_price +
    p[3][3] * cave_tour_price +
    p[3][4] * museum_tour_price +
    p[3][5] * city_tour_price +
    p[3][6] * garden_tour_price
)

print('The best combo is ', f"Park: {p[3][0]} "+
                                   f"Pool: {p[3][1]} "+
                                   f"Castle: {p[3][2]} "+
                                   f"Cave: {p[3][3]} "+
                                   f"Museum: {p[3][4]} "+
                                   f"City: {p[3][5]} "+
                                   f"Garden: {p[3][6]}")
# If 1 is printed, the activity is within the best budget combo. If 0 is printed, it is not within the best budget combo.

print(f"Total Cost of the Best Combo: {best_combo_cost}")
print(f"Spare Change: {p[0].min()}")          # p[0] is the spare_change_values 
print('-------')
print(f"Failed combinations and their costs:")
for combo, cost in zip(p[4], p[5]):           # p[4] is the failed combinations and p[5] is their costs 
    print(f"Combo: Park: {combo[0]}, Pool: {combo[1]}, Castle: {combo[2]}, Cave: {combo[3]}, Museum: {combo[4]}, City: {combo[5]}, Garden: {combo[6]} - Cost: {cost}")

print('-------')
```

#### Adding the Time Loop to the Main Code

Using the activities printed from best combo, it is now time to add the time loop. This will tell you what activities you have time for on your vacation. It will then tell you which activites can be done on each day given and how much you are spending each day of vacation and your new spare change total if an activity was not within the time restraints. As shown in this example, the castle tour was excluded and the money was put back in the budget. 

```python
# AFTER DETERMINING THE BEST COMBO FOR COST ADD THE TIME LOOP WITH ONLY THE ACTIVITIES IT SAID WAS IN BUDGET

# Check if activities can be done in time of vacation 

# Imput variables
t = vpl.time(national_park_time, castle_tour_time, cave_tour_time, museum_tour_time, garden_tour_time, max_time)

best_combo_time = (
    t[3][0] * national_park_time +
    t[3][1] * castle_tour_time +
    t[3][2] * cave_tour_time + 
    t[3][3] * museum_tour_time + 
    t[3][4] * garden_tour_time
)

print('The best time combo with activities from best budget is ', f"Park: {t[3][0]} "+
                                   f"Castle: {t[3][1]} "+
                                   f"Cave: {t[3][2]} "+
                                   f"Museum: {t[3][3]} "+
                                   f"Garden: {t[3][4]}")
# If 1 is printed, the activity is within the best time combo. If 0 is printed, it is not within the best time combo.

print(f"Total Time of the Best Combo: {best_combo_time}")
print(f"Spare Time: {t[0].min()}")
print('-------')
print(f"Failed combinations and their times:")
for combo, time in zip(t[4], t[5]):
    print(f"Combo: Park: {combo[0]}, Castle: {combo[1]}, Cave: {combo[2]}, Museum: {combo[3]}, Garden: {combo[4]} - Time: {time}")

print('-------')

# List of activities per day 
activities_day_1 = []
activities_day_2 = []
time_for_day_1 = 8   # in hours 
time_for_day_2 = 8

activity_times = [national_park_time, castle_tour_time, cave_tour_time, museum_tour_time, garden_tour_time]
activity_names = ["Park", "Castle", "Cave", "Museum", "Garden"]
activity_prices = [national_park_price, castle_tour_price, cave_tour_price, museum_tour_price, garden_tour_price]

day_1_cost = 0 
day_2_cost = 0 

for i, activity in enumerate(t[3]):
	if activity == 1:
		if activity_times[i] <= time_for_day_1:
			activities_day_1.append(activity_names[i])
			time_for_day_1 -= activity_times[i]
			day_1_cost += activity_prices[i]
		elif activity_times[i] <= time_for_day_2:
			activities_day_2.append(activity_names[i])
			time_for_day_2 -= activity_times[i]
			day_2_cost += activity_prices[i]

total_spare_change = max_money - (day_1_cost + day_2_cost)

print(f'Activities for Day 1: {activities_day_1}')
print(f'Cost for Day 1: {day_1_cost}')
print(f'Activities for Day 2: {activities_day_2}')
print(f'Cost for Day 2: {day_2_cost}')
print(f'Total Spare Change for Trip: {total_spare_change}')
```
### Contributions 

If you see a way to improve the code please let me know. Detail what you see can be changed and how it compares to the original code. 

### Acknowledgments 

I would like to thank my professor Dr. Joyce for teaching me all I learned for coding. I never thought I would be writing a program one day, even if it is a small one. 
