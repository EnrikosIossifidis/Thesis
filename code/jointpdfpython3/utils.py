from collections import Sequence
import funcy
import itertools

def flatten(alist):
    return list(funcy.flatten(alist))


def maximum_depth(seq):
    """
    Helper function, e.g. maximum_depth([1,2,[2,4,[[4]]]]) == 4.
    :param seq: sequence, like a list of lists
    :rtype: int
    """
    seq = iter(seq)
    try:
        for level in itertools.count():
            seq = itertools.chain([next(seq)], seq)
            seq = itertools.chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level


# helper function,
# from http://stackoverflow.com/questions/2267362/convert-integer-to-a-string-in-a-given-numeric-base-in-python
def int2base(x, b, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    """

    :param x: int
    :type x: int
    :param b: int
    :param b: int
    :param alphabet:
    :rtype : str
    """

    # convert an integer to its string representation in a given base
    if b < 2 or b > len(alphabet):
        if b == 64: # assume base64 rather than raise error
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        else:
            raise AssertionError("int2base base out of range")

    if isinstance(x,complex): # return a tuple
        return int2base(x.real,b,alphabet), int2base(x.imag, b, alphabet)

    if x <= 0:
        if x == 0:
            return alphabet[0]
        else:
            return '-' + int2base(-x, b, alphabet)

    # else x is non-negative real
    rets = ''

    while x>0:
        x,idx = divmod(x,b)
        rets = alphabet[idx] + rets

    return str(rets)

def get_var_labels(n_vars):
    vs = ['X{}'.format(i) for i in range(n_vars-1)]
    vs.append('Y')
    return vs

def apply_permutation(lst, permutation):
    """
    Return a new list where the element at position ix in <lst> will be at a new position permutation[ix].
    :param lst: list
    :type lst: array_like
    :param permutation:
    :return:
    """
    new_list = [-1]*len(lst)

    assert len(permutation) == len(lst)

    for ix in range(len(permutation)):
        new_list[permutation[ix]] = lst[ix]

    if __debug__:
        if -1 not in lst:
            assert -1 not in new_list

    return new_list
