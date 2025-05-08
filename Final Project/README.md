# Budgeting and Timing a Vacation

This repository optimizes a resource with given restraints. In this case, the resource is the budget and the restraints are prices and time. 

## Table of Contents 

1. [Description](#description)
2. [Installation](#installation)
3. [Useage](#useage)

## Description 

This project is made to help someone on a budget going on vacation. With a budget set aside to do activities, one can max their experience by doing as many activities as possible. After setting a budget, and seeing what activities would be best, those activities are then put into a similar loop but for time restraints instead of money. This part tells you what activities are within your vacation time set aside for activities. This code is not to help you save money put to maximize its use. 

## Installation 

For this code to work, you need to download the vacation_project_lib.py and import it into the Final_Project.py file. You also need to import numpy. To make sure the code works properly, the cost loop needs to run first before putting the timing loop after it. From the cost loop, use the answer given to know what activities will be needed for the time loop. Once everything is set, they can run together. 

### Vacation Library 

In the vacation_project_lib.py, a price and time function are given. These need to be changed to your own activities, prices, and times. If you are adding activities, or sometime just changing the ones given, an error may appear that says you forgot a comma. You did not. To fix this you will need to rewrite the variable it is having trouble with. In the library file this will happen at total cost.\
The varaibles in the price function are the prices of each activity and the max money you are willing to spend. Similarly, the time function variables are the times, in hours, of each activity and the max time you can spend doing activities. The for loop makes sure each activity can only be done once or not at all. If you want to see if you can do an activity more than once, you will need to change the range from 2 to 3.  

### Final Project Code 

Once your library is set up, it needs to be imported to the .py file. Then, make a list of the prices and times of each activity. State what your max money and time is, then make it a float. If desirable, the time can be changed from hours to minutes. Next, call on the library you made to find what activities fit in your budget best and filling in the variables. To find the best combo of activities, the activity count in best combo needs to be multiplied by the price of the activity. Do this for each activity and add them together. 
