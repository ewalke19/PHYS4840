#!/usr/bin/env python3.8
#
# Author: Erin Walker
#
#########################
# 
# Possible Combos for Each Activity Without Repeating One. Modify as needed for extra activities.  


import vacation_project_lib as vpl 
import numpy as np 

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

# YOU HAVE TO RUN THE COST LOOP FIRST THEN ADD THE TIME LOOP

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

# AFTER DETERMINING THE BEST COMBO FOR COST ADD THE TIME LOOP WITH ONLY THE ACTIVITIES IT SAID WAS IN BUDGET

# Check if activities can be done in time of vacation 

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
# If 1 is printed, the activity is within the best budget combo. If 0 is printed, it is not within the best budget combo.

print(f"Total Time of the Best Combo: {best_combo_time}")
print(f"Spare Time: {t[0].min()}")
print('-------')
print(f"Failed combinations and their times:")
for combo, time in zip(t[4], t[5]):
    print(f"Combo: Park: {combo[0]}, Castle: {combo[1]}, Cave: {combo[2]}, Museum: {combo[3]}, Garden: {combo[4]} - Time: {time}")


# Calculate acvitities per day 
#activities_per_day = best_combo_time/2 # 2 is the number of days 
#print(f'Time per day: {activities_per_day}')

print('-------')

# List of activities per day 
activities_day_1 = []
activities_day_2 = []
time_for_day_1 = 8
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







