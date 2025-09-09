from psychopy import visual, event
from psychopy.visual import ElementArrayStim
from sklearn.gaussian_process.kernels import Kernel
import numpy as np
import math, time, random


def initialize_psychopy(fullscr=False):
    """
    Initialize PsychoPy visual elements and return dictionary of them.
    
    Args: None
    Returns: Dictionary of PsychoPy visual components
    """
    win = visual.Window([1800, 1200], color='grey', units='pix', fullscr=fullscr)
    line = visual.Line(win, start=(-500, 0), end=(500, 0), lineColor='black')
    line_leftend = visual.Line(win, start=(-500, -10), end=(-500, 10), lineColor='black')
    line_rightend = visual.Line(win, start=(500, -10), end=(500, 10), lineColor='black')
    left_box = visual.Rect(win, width=300, height=300, pos=(-500, -200), fillColor=None, lineColor='black')
    right_box = visual.Rect(win, width=300, height=300, pos=(500, -200), fillColor=None, lineColor='black')
    question_box = visual.Rect(win, width=300, height=300, pos=(0, 200), fillColor=None, lineColor='black')
    marker = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='orange', lineWidth=3)
    img_stim = visual.ImageStim(
        win=win,
        image='./nlt_main/images/random_noise.png',
        pos=(0, 200),
        size=(300, 300)
    )
    visuals = {'win': win,
                'line': line,
                'line_leftend': line_leftend,
                'line_rightend': line_rightend,
                'left_box': left_box,
                'right_box': right_box,
                'question_box': question_box,
                'marker': marker,
                'img_stim': img_stim
                }   

    return visuals

def draw_grid_position(n_dots, dot_size, boxW, boxH, box_center=(0, 0), padding=10):
    """
    Draws a grid of positions for the dots within a specified box, 
    so that the dots are evenly spaced.

    Args:
    - n_dots: number of dots
    - dot_size: size(diameter) of dots
    - boxW, boxH: dimensions of the box (width, height)
    - box_center: center position of the box (x, y)
    - padding: padding between dots and box edges

    Returns:
    - jittered_positions: list of tuples representing the (x, y) positions of the dots
    """
    itemWH = dot_size * 1.75
    usable_boxW = boxW - 2 * (padding + dot_size / 2)
    usable_boxH = boxH - 2 * (padding + dot_size / 2)

    X = int(usable_boxW // itemWH)
    Y = int(usable_boxH // itemWH)

    if X * Y < n_dots:
        raise ValueError("Not enough grid positions for the number of dots.")
    
    grid_x = np.linspace(-usable_boxW / 2, usable_boxW / 2, X)
    grid_y = np.linspace(-usable_boxH / 2, usable_boxH / 2, Y)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_positions = list(zip(grid_X.flatten(), grid_Y.flatten()))
    random.shuffle(grid_positions)

    selected_positions = grid_positions[:n_dots]

    jittered_positions = []
    jitter_ratio = 0.3
    for x, y in selected_positions:
        jitter_x = random.uniform(-itemWH * jitter_ratio, itemWH * jitter_ratio)
        jitter_y = random.uniform(-itemWH * jitter_ratio, itemWH * jitter_ratio)
        final_x = x + jitter_x + box_center[0]
        final_y = y + jitter_y + box_center[1]
        jittered_positions.append((final_x, final_y))

    return jittered_positions

def calculate_dot_size(max_number_size, number, max_number=500):
    """
    Calculate the size of the dots based on the number of dots.
    Size of the given number of dots is calculated to be same as the cumulative area of the upperbound dots.

    Args:
    - max_number_size: maximum size(diameter) of the dots
    - number: number of dots
    - max_number: maximum number of dots

    Returns:
    - dot_size: size(diameter) of the dots
    """
    original_r = max_number_size / 2
    total_upper_area = math.pi * (original_r ** 2) * max_number

    current_dot_area = total_upper_area / number
    current_r = math.sqrt(current_dot_area / math.pi)

    stimdot_size1 = math.floor(current_r * 2)
    stimdot_size2 = math.ceil(current_r * 2)

    error1 = abs(total_upper_area - (stimdot_size1 / 2) ** 2 * math.pi * number)
    error2 = abs(total_upper_area - (stimdot_size2 / 2) ** 2 * math.pi * number)

    stimdot_size = stimdot_size1 if error1 < error2 else stimdot_size2
    return stimdot_size

def draw_base_components(visuals):
    """
    Draw base components (line, left_box, right_box, leftend, rightend, question_box) on win.

    Args:
    - win: PsychoPy window object
    - visuals: Dictionary of PsychoPy visual components
    """

    visuals['left_box'].draw()
    visuals['right_box'].draw()
    visuals['question_box'].draw()
    visuals['line'].draw()
    visuals['line_leftend'].draw()
    visuals['line_rightend'].draw()

def show_and_get_response(number, visuals, max_number, size_control=False):
    """
    Show and get response from the user.

    Args:
    - number: number of dots
    - visuals: dictionary of PsychoPy visual components to show
    - max_number: maximum number of dots
    - size_control: whether to control the size of the dots

    Returns:
    - response: the user's response
    """
    win = visuals['win']
    marker = visuals['marker']
    img_stim = visuals['img_stim']

    if isinstance(number, float):
        number=int(number)
    if isinstance(max_number, float):
        max_number=int(max_number)
    if size_control:
        dot_size = calculate_dot_size(max_number_size=4, number=number, max_number=max_number)
    else:
        dot_size = 4

    # Draw dots & base visual components
    right_dots_pos = draw_grid_position(max_number, 4, 300, 300, box_center=(500, -200), padding=25)
    right_dots = ElementArrayStim(win,
                                   nElements=max_number,
                                   xys=right_dots_pos,
                                   sizes=4,
                                   colors='orange',
                                   elementTex=None,
                                   elementMask='circle',
                                   interpolate=True)

    question_dots_pos = draw_grid_position(number, dot_size, 300, 300, box_center=(0, 200), padding=25)
    question_dots = ElementArrayStim(win,
                                      nElements=number,
                                      xys=question_dots_pos,
                                      sizes=dot_size,
                                      colors='orange',
                                      elementTex=None,
                                      elementMask='circle',
                                      interpolate=True)
    draw_base_components(visuals)
    right_dots.draw()
    question_dots.draw()
    win.flip()
    start_time = time.time()

    # Defining mouse click flag
    clicked = False
    
    while not clicked:
        if time.time() - start_time > 2:
            # After 2 seconds, mask the given dots box
            img_stim.draw()
            right_dots.draw()
            draw_base_components(visuals)
            win.flip()

        # Waiting for mouse click
        mouse = event.Mouse(win=win)
        if mouse.getPressed()[0]:
            x, y = mouse.getPos()

            # If the click is outside the valid area, ignore it
            if x < -500 or x > 500 or y < -20 or y > 20:
                continue

            # If the click is valid, record the position and show the marker
            clicked = True
            marker.pos = (x, 0)
            marker.draw()
            win.flip()

    # Draw the response position
    visuals['line'].draw()
    visuals['line_leftend'].draw()
    visuals['line_rightend'].draw()
    visuals['left_box'].draw()
    visuals['right_box'].draw()
    right_dots.draw()
    marker.draw()
    win.flip()

    # Return the participant's response (estimation value)
    return int((x + 500) / 1000 * max_number)
