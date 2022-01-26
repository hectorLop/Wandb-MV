def smaller_than(x, y):
    return x < y

def smaller_or_equal(x, y):
    return x <= y

def greater_than(x, y):
    return x > y

def greater_or_equal(x, y):
    return x >= y

COMP_FUNC = {
    'smaller': smaller_than,
    'smaller_or_equal': smaller_or_equal,
    'greater': greater_than,
    'greater_or_equal': greater_or_equal
}

def compare(x, y, mode):
    if not mode in COMP_FUNC:
        raise ValueError(f'There aren\'t any function to {mode} comparisions')

    return COMP_FUNC[mode](x, y)