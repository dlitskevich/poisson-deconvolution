from functools import lru_cache
from itertools import combinations


# very slow for n>10
def setpartition(iterable, n=2):
    """
    Generate all set partitions of an iterable into n groups.

    Parameters:
    iterable (iterable): The iterable to partition.
    n (int): The number of elemets in each group.

    Returns:
    generator: A generator of all set partitions of the iterable into groups of size n.
    """
    iterable = list(iterable)
    partitions = combinations(combinations(iterable, r=n), r=len(iterable) // n)
    for partition in partitions:
        seen = set()
        for group in partition:
            if seen.intersection(group):
                break
            seen.update(group)
        else:
            yield partition


@lru_cache(maxsize=None)
def count_partitions_binomial(n: int, k: int) -> dict[tuple, int]:
    """
    Count the number of partitions of n into distinctive pairs with k ones.

    Args:
        n: The number to partition.
        k: The number of ones.
    """
    if n == 2:
        match k:
            case 0:
                return {(0,): 1}
            case 1:
                return {(1,): 1}
            case 2:
                return {(2,): 1}
    array = [0 for _ in range(n - k)] + [1 for _ in range(k)]
    partitions: dict[tuple, int] = {}
    for i in range(1, n):
        mix = array[i] + array[0]
        inner_partitions = count_partitions_binomial(n - 2, k - mix)
        for p, c in inner_partitions.items():
            partition = tuple(sorted((mix,) + p))

            partitions[partition] = partitions.get(partition, 0) + c

    return partitions
