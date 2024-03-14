# This file is meant to scrape a website to get the live occupancy of McComas Gym
#import numpy as np
from datetime import datetime, timezone, timedelta
import schedule
import time as timer
import csv
from csv import writer
import requests


def recordCapData():
  res = requests.get('https://connect.recsports.vt.edu/facilityoccupancy')

  fullText = res.text
  stat = res.status_code
  if stat == 200:
    splitted = fullText.split('\n')

    #the line number of where the occupancy is mentioned - 573 is the line number
    spot = len(splitted) - 286
    #print(spot)
    #the full line of code that containes the current occupancy
    line = splitted[spot]
    #the line with the occupacy starting at index 0
    updatedLine = line[150:]
    #save current occupancy
    currentOccupancy = ''
    currIndex = 0
    #save the current occupancy
    while(updatedLine[currIndex].isdigit()):
      currentOccupancy += updatedLine[currIndex]
      currIndex += 1

    print("Current Occupancy: ", currentOccupancy)

    #get day of week and time of recording
    tzinfo = timezone(timedelta(hours=-4))  #timezone offset for EST is -5
    #use to get time and day of week
    date = datetime.now(tzinfo)
    # Monday is 0, sunday is 6 - use isoweekday() for Monday is 1, Sunday is 7
    day_of_week = date.weekday()
    # current time - 0:00-23:59 EST
    time = date.time()
    # current date
    c_date = date.date()

    if day_of_week == 5 or day_of_week == 6:
      startTime = datetime(year=c_date.year, month=c_date.month, day=c_date.day, hour = 10, minute = 0, second = 0, tzinfo = tzinfo)
      endTime = datetime(year=c_date.year, month=c_date.month, day=c_date.day, hour = 22, minute = 0, second = 0, tzinfo = tzinfo)
    else:
      startTime = datetime(year=c_date.year, month=c_date.month, day=c_date.day, hour = 6, minute = 0, second = 0, tzinfo = tzinfo)
      endTime = datetime(year=c_date.year, month=c_date.month, day=c_date.day, hour = 23, minute = 0, second = 0, tzinfo = tzinfo)

    print("Current Time: ", date)
    #print(c_date)
    #print(time)
    #print(day_of_week)
    print()

    if time > startTime.time() and time < endTime.time():
      # data list containing datetime values
      data = [c_date, time, day_of_week]
      # target list containing current occupancy
      target = [currentOccupancy, c_date, time]

      # Open existing trainingData csv file and save date, time, and day of week
      with open('mccomasTarget.csv', 'a', newline='') as targetFile:

        # Pass this file object to csv.writer()
        # and get a writer object
        write = writer(targetFile)

        # Pass the list as an argument into
        # the writerow()
        write.writerow(target)

        # Close the file object
        targetFile.close()

      # Open existing trainingData csv file and save date, time, and day of week
      with open('mccomasData.csv', 'a', newline='') as trainingFile:

        # Pass this file object to csv.writer()
        # and get a writer object
        write = writer(trainingFile)

        # Pass the list as an argument into
        # the writerow()
        write.writerow(data)

        # Close the file object
        trainingFile.close()
    else:
      print("McComas is not currently open.")
  else:
    print("=========================") 
    print("Could NOT access Website.")
    print("=========================")

# run function every 10 minutes
schedule.every(20).minutes.do(recordCapData)
recordCapData()
while True:
  # runs recordCapData function every x minutes
  schedule.run_pending()
  # run while loop once every second
  timer.sleep(1)
#recordCapData()