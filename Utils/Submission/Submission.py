import RootPath


def create_submission_file(tweets, users, predictions, output_file):

    # Preliminary checks
    assert len(tweets) == len(users)
    assert len(users) == len(predictions)
    assert len(tweets) == len(predictions)

    file = open(RootPath.get_root().joinpath(output_file), "w")

    for i in range(len(tweets)):
        file.write(f"{tweets[i]},{users[i]},{predictions[i]}\n")

    file.close()