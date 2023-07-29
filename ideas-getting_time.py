import datetime

#Here we can see an example program to filter information within a specific timeframe

#Start: 1512691200
#End:  1512777599

dt1 = datetime.datetime.fromtimestamp(1512691200)

date1 = (dt1.year, dt1.month, dt1.day, dt1.hour)

print(date1)

dt2 = datetime.datetime.fromtimestamp(1512777599)

date2 = (dt2.year, dt2.month, dt2.day, dt2.hour)

print(date2)

#Given the start and end dates we find the articles/text related to summarize

#Here will be the list of events from the dowloaded data 
eventlist = []
#Here we will have the days for the particular event of interest
daysInEvent = []

for day in daysInEvent:
    if dt1.day == day or dt2.day == day:
        #Here we will gather all the text related to that date for summarization
        print("What happened in ", day)