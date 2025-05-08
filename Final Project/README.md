# Budgeting and Timing a Vacation

This repository optimizes a resource with given restraints. In this case, the resource is the budget and the restraints are prices and time. 

## Table of Contents 

1. [Description](#description)
2. [Installation](#installation)
3. [Useage](#useage)

## Description 

This project is made to help someone on a budget going on vacation. With a budget set aside to do activities, one can max their experience by doing as many activities as possible. After setting a budget, and seeing what activities would be best, those activities are then put into a similar loop but for time restraints instead of money. This part tells you what activities are within your vacation time set aside for activities. 

## Installation 

For this code to work, you need to download the vacation_project_lib.py and import it into the Final_Project.py file. You also need to import numpy. To make sure the code works properly, the cost loop needs to run first before putting the timing loop after it. From the cost loop, use the answer given to know what activities will be needed for the time loop. Once everything is set, they can run together. 

### Vacation Library 

In the vacation_project_lib.py, a price and time function are given. These need to be changed to your own activities, prices, and times. If you are adding activities, or sometime just changing the ones given, an error may appear that says you forgot a comma. You did not. To fix this you will need to rewrite the variable it is having trouble with. In the library file this will happen at total cost. 
