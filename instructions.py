import json
import functools


status_dict = {"in process": 3, "waiting":2, "completed":1}

def compare (r1, r2):
    s1 = r1["status"]
    s2 = r2["status"]
    s1p = status_dict.get(s1, 0)
    s2p = status_dict.get(s2, 0)

    if s1p < s2p:
        return 1
    if s1p > s2p:
        return -1

    p1 = r1["priority"]
    p2 = r2["priority"]

    if p1 < p2:
        return 1
    if p1 > p2:
        return -1

    n1 = r1["dso"]
    n2 = r2["dso"]

    if n1 < n2:
        return 1
    if n1 > n2:
        return -1

    return 0








def get_dso_object_tonight():
    with open('instructions.json', 'r') as f:
        instructions = json.load(f)

    sorted_l = sorted(instructions, key=functools.cmp_to_key(compare))
    best = sorted_l[0]

    return best



if __name__ == "__main__":
    best = get_dso_object_tonight()
    print(best)

