import os

def flatten(l):
    return [item for sublist in l for item in sublist]


def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert (len({len(it) for it in list_of_iterables}) == 1)
    l = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, l, n):
            if not return_incomplete_batches:
                if (ndx + n) > l:
                    break
            yield [iterable[ndx:min(ndx + n, l)] for iterable in list_of_iterables]

        if not infinite:
            break


def recursive_listdir(path):
    files = []
    for dirName, subdirList, fileList in os.walk(path):
        for fname in fileList:
            filepath = os.path.join(dirName, fname)
            yield filepath
