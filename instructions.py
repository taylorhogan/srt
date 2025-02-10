import json
import functools

def compare (r1, r2):
    print (r1)
    print (r2)
    return -1

def get_dso_object_tonight():
    with open('instructions.json', 'r') as f:
        instructions = json.load(f)
    requests =instructions['requests']
    sorted_l = sorted(requests, key=functools.cmp_to_key(compare))
    best = sorted_l[0]
    return best



if __name__ == "__main__":
    best = get_dso_object_tonight()
    print(best)

