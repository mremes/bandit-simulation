def container_for(k: int, default_value=0):
    return dict.fromkeys(map(lambda x: str(x), range(1, k+1)), default_value)
