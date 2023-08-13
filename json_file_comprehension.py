import json 

with open("C:/Users/aarya/Downloads/CrisisFACTs-2022.facts.json", "r", encoding = "utf-8") as j_file:
    data = json.load(j_file)
    
file1 = open("output.txt", "w", encoding="utf-8")
counter = 1

def time_analysis(list_times, current_time):
    for key in list_times:
        times = list_times[key] 
        if current_time >= times[0] and current_time <= times[1]:
            return key
    return str(list_times    )
        

for x in data: 
    
    summaryRequests = x["summaryRequests"]
    dictionary = {}
    for request in summaryRequests:
        dictionary[request["requestID"]] = [request["startUnixTimestamp"]*1000, request["endUnixTimestamp"]*1000]
        
    
    
    file1.write("EVENT NUMBER ")
    file1.write(str(counter))
    file1.write("\n\n\n\n\n")
    new_list = []
    y = x["factsByRequest"]
    for x in y:
        z = y[x]
        for obj in z:
           # print(obj["fact"])
            fact = obj["fact"]
            
            file1.write(fact)
            
            file1.write("\n associated request ID: ")
            
             
            file1.write(time_analysis(dictionary, obj["dateUnix"]))
            
            file1.write("\n")
            
    counter = counter + 1