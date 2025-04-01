import json
import functools
from venv import create

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from astropy.time import Time


import dso_visibility

import utils

status_dict = {"in process": 3, "waiting":2, "completed":1}

def calc_and_store_hours_above_horizon ():
    utils.set_install_dir()
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        dso = text = instruction["dso"]
        obj = dso_visibility.is_a_dso_object(dso)
        if obj is not None:
            above = dso_visibility.get_above_horizon_time(obj, Time.now())
            instruction["above_horizon"] = str(above)
        else:
            instruction["above_horizon"] = '0'
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


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

    #oh1 = r1["above_horizon"]
   #oh2 = r2["above_horizon"]

    n1 = r1["dso"]
    n2 = r2["dso"]

    if n1 < n2:
        return 1
    if n1 > n2:
        return -1

    return 0






def create_instructions_table ():
    calc_and_store_hours_above_horizon()
    sorted_l = get_sorted_instructions()
    row_idx =  len(sorted_l) + 1

    fig, ax = plt.subplots(figsize=(10,row_idx + 4))
    ax.axis('off')

    # Set the title
    #ax.set_title("Image Requests", fontsize=20, pad=20)

    # Weekday labels
    headers = ['DSO', 'Requestor', 'State', 'Date', 'Tonight']
    for i, header in enumerate(headers):
        ax.text(i + 2.5, len(sorted_l) + 1, header, ha='center', fontsize=12, weight='bold')

    # Generate the  grid

    for instruction in sorted_l:
       # Get color and text for the day
        color='gray'
        text = 'development'
        if instruction["status"] == 'in process':
            color = 'lightgreen'
        if instruction["status"] == 'waiting':
            color = 'lightblue'
        if instruction["status"] == 'completed':
            color = 'pink'
        for col_idx in range (5):
            text = ''
            if col_idx == 0:
                text = instruction["dso"]
            elif col_idx == 1:
                text = instruction["requestor"]
            elif col_idx == 2:
                text = instruction["status"]
            elif col_idx == 3:
                string = instruction["request_time"]
                if string != "":
                    datetime_object = datetime.strptime(string, '%Y-%m-%d')
                    formatted_date = datetime_object.strftime("%m\n%d\n%Y")
                    text = formatted_date
                else:
                    text = ""
            elif col_idx == 4:
                text = instruction["above_horizon"]


            rect = mpatches.Rectangle((col_idx+2, row_idx-1), 1, 1, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            # Add text
            ax.text(col_idx + 2.5, row_idx - 0.5, text, ha='center', va='center', fontsize=12)

        row_idx = row_idx -1
        #break


    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    #ax.set_aspect('equal')
    fig.savefig ('instructions.png')


def add_dso_object_instruction (dso_name, recipe, requestor, priority=0):
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    new_instruction = {
      "dso": dso_name,
      "uuid": "1",
      "recipe": recipe,
      "requestor": requestor,
      "request_time": formatted_date,
      "status": "waiting",
      "priority": priority
    }
    utils.set_install_dir()
    instructions.append(new_instruction)
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def get_sorted_instructions():
    utils.set_install_dir()

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






