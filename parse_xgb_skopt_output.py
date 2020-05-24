def parse(name):
    file = open(name)
    out = open(f"{name}.csv", "w")

    print(name)

    first_line = [0]
    ignored_lines = [0, 14, 16, 18, 21, 26, 31, 35, 39, 41]
    new_lines = [42]
    objective_lines = [15, 17, 40]
    lines = 43

    for i, line in enumerate(file.readlines()):

        if (i % lines) in ignored_lines:
            pass
        elif (i % lines) in objective_lines:
            value = line[:-1].split(":")[0].strip(" ")
            out.write(f"{value},")
        elif (i % lines) in new_lines:
            out.write(f"\n")
        else:
            value = line[:-1].split("=")[0].strip(" ")
            out.write(f"{value},")

        if i >= lines:
            break

    file = open(name)
    for i, line in enumerate(file.readlines()):
        print(i)
        if (i % lines) in ignored_lines:
            pass
        elif (i % lines) in objective_lines:
            value = line[:-1].split(":")[1].strip(" ")
            out.write(f"{value},")
        elif (i % lines) in new_lines:
            out.write(f"\n")
        else:
            value = line[:-1].split("=")[1].strip(" ")
            out.write(f"{value},")
    out.flush()
    out.close()

if __name__ == '__main__':

    parse("like")
    parse("retweet")
    parse("comment")
    parse("reply")
