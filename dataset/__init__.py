from .manager import Community, Ego, Ladders, TUData


def get_dataset_class(name):
    if name.lower() == 'community':
        return Community

    if name.lower() == 'ego':
        return Ego

    if name.lower() == 'ladders':
        return Ladders

    return TUData
