if __name__ == '__main__':
    file = open("retweet.log")
    out = open("retweet.log.csv", "w")

    first_line = [0]
    ignored_lines = [0, 14, 16, 19, 24, 29, 33, 37, 39]
    new_lines = [40]
    objective_lines = [15, 38]
    lines = 41

    for i, line in enumerate(file.readlines()):

        if (i % lines) in ignored_lines:
            pass
        elif (i % lines) in objective_lines:
            value = line[:-1].split(":")[0].strip(" ")
            out.write(f"{value}")
        elif (i % lines) in new_lines:
            out.write(f"\n")
        else:
            value = line[:-1].split("=")[0].strip(" ")
            out.write(f"{value},")

        if i >= lines:
            break

    file = open("retweet.log")
    for i, line in enumerate(file.readlines()):
        print(i)
        if (i % lines) in ignored_lines:
            pass
        elif (i % lines) in objective_lines:
            value = line[:-1].split(":")[1].strip(" ")
            out.write(f"{value}")
        elif (i % lines) in new_lines:
            out.write(f"\n")
        else:
            value = line[:-1].split("=")[1].strip(" ")
            out.write(f"{value},")
    out.flush()
    out.close()