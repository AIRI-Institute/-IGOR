def check_format_correctness(subtask):
    try:
        subtask_tuple = eval(subtask)
        if len(subtask_tuple) != 4:
            return False
    except:
        return False
    all_digit = sum([isinstance(s, int) for s in subtask_tuple])
    return all_digit==4