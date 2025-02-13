import json
import functools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

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






def create_instructions_table ():
    sorted_l = get_sorted_instructions()
    row_idx = len(sorted_l) + 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Set the title
    ax.set_title("Image Requests", fontsize=20, pad=20)

    # Weekday labels
    headers = ['DSO', 'Requestor', 'State', 'Date']
    for i, header in enumerate(headers):
        ax.text(i + 0.5, row_idx+ 1, header, ha='center', fontsize=12, weight='bold')

    # Generate the  grid

    for instruction in sorted_l:



        # Get color and text for the day
        color='gray'
        if instruction["status"] == 'in process':
            color = 'lightgreen'
        if instruction["status"] == 'waiting':
            color = 'lightblue'
        if instruction["status"] == 'completed':
            color = 'pink'
        for col_idx in range (4):
            text = ''
            if col_idx == 0:
                text = instruction["dso"]
            elif col_idx == 1:
                text = instruction["requestor"]
            elif col_idx == 2:
                text = instruction["status"]
            elif col_idx == 3:
                text = instruction["request_time"]


            rect = mpatches.Rectangle((col_idx, row_idx), 1, 1, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            # Add text
            ax.text(col_idx + 0.5, row_idx + 0.5, text, ha='center', va='center', fontsize=12)

        row_idx = row_idx -1



    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')

    plt.show()
    fig.savefig ('instructions.png')

def add_dso_object_instruction (dso_name, recipe, requestor, priority=0):
    now = datetime.now()
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    new_instruction = {
      "dso": dso_name,
      "uuid": "1",
      "recipe": recipe,
      "requestor": requestor,
      "request_time": str(now),
      "status": "waiting",
      "priority": priority
    }
    instructions.append(new_instruction)
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def get_sorted_instructions():
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)

    sorted_l = sorted(instructions, key=functools.cmp_to_key(compare))
    return sorted_l



def get_dso_object_tonight():
    sorted_l =get_sorted_instructions()

    best = sorted_l[0]

    return best



if __name__ == "__main__":
    create_instructions_table()
    #add_dso_object_instruction("foo", "mayo", "teh", 0)



