import calendar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_custom_calendar(year, month, day_colors_text):
    # Generate the calendar for the given month
    cal = calendar.Calendar()
    days = cal.itermonthdays2(year, month)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Set the title
    ax.set_title(calendar.month_name[month] + f" {year}", fontsize=20, pad=20)

    # Weekday labels
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    #for i, weekday in enumerate(weekdays):
       # ax.text(i + 0.5, 6.5, weekday, ha='center', fontsize=12, weight='bold')

    # Generate the calendar grid
    row = 6
    col = 7
    for (day, weekday) in days:
        if day == 0:
            continue  # Skip days not in the month

        week_row = row - (day + weekday - 1) // 7
        col_pos = weekday

        # Get color and text for the day
        color, text = day_colors_text.get(day, ("white", str(day)))

        # Draw day cell
        rect = mpatches.Rectangle((col_pos, week_row), 1, 1, edgecolor="black", facecolor=color)
        ax.add_patch(rect)

        # Add text
        ax.text(col_pos + 0.5, week_row + 0.5, text, ha='center', va='center', fontsize=12)

    # Set limits and aspect
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)
    ax.set_aspect('equal')

    plt.show()

# Example usage
year = 2025
month = 1
day_colors_text = {
    1: ("lightblue", "Holiday"),
    7: ("lightgreen", "Meeting"),
    15: ("yellow", "Payday"),
    25: ("orange", "Event"),
    31: ("pink", "Deadline")
}

generate_custom_calendar(year, month, day_colors_text)