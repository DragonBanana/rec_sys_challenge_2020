from Utils.TelegramBot import telegram_bot_send_update

import requests

import time

r = requests.get("https://recsys-twitter.com/leaderboard/latest")

result = str(r.content)

groupname = "12345678."
start_index = result.find('12345678.')

print(start_index)

toparse = result[start_index:start_index+350]

print(toparse)

splitted = toparse.split("<td>")

for i in range(0,10):
    splitted[i] = splitted[i][0:splitted[i].index('<')]
    print(splitted[i])

praucs= [splitted[2], splitted[4], splitted[6], splitted[8]]
rces = [splitted[3], splitted[5], splitted[6], splitted[9]]

classes = ["retweet","reply", "like", "rt with comment"]

for c in classes:
    i = 0
    print(f"{c}: PRAUC:\t{praucs[i]}, RCE:\t{rces[i]}")

while True:
    r = requests.get("https://recsys-twitter.com/leaderboard/latest")

    result = str(r.content)

    groupname = "12345678."
    start_index = result.find('12345678.')

    print(start_index)

    toparse = result[start_index:start_index+350]

    print(toparse)

    splitted = toparse.split("<td>")

    for i in range(0,10):
        splitted[i] = splitted[i][0:splitted[i].index('<')]
        print(splitted[i])

    new_praucs= [splitted[2], splitted[4], splitted[6], splitted[8]]
    new_rces = [splitted[3], splitted[5], splitted[6], splitted[9]]

    for i in range(0,4):
        if new_praucs[i] != praucs[i]:
            telegram_bot_send_update(f"{classes[i]}:\n PREVIOUS RESULTS: \nPRAUC:\t{praucs[i]}, RCE:\t{rces[i]}\n NEW RESULTS: \nPRAUC:\t{new_praucs[i]}, RCE:\t{new_rces[i]}")
            praucs[i] = new_praucs[i]
            rces[i] = new_rces[i]

    time.sleep(10)