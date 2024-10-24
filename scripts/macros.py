import time
import keyboard

'''
handles macros for me to play elden ring
'''
# key = action, value = button that action is bound to
bindings:dict={
    'Forward': 'w',
    'Left': 'a',
    'Back': 's',
    'Right': 'd',

    'Camera': 'q', 
    'Heal': 'f',

    'Attack': 'p', 
    'Roll': 'left shift',

    'Sprint': 'left shift',

    'Stop': 'stop'

}
prevAction = "Stop" # last action

def input(gesture:str, hand:str = 'right')->None:
    '''
    Converts given gesture to an action that will be performed in game

    Actions
        Roll, Attack, Heal
            will be tapped repeatedly as long as the corresponding gesture is held up

        Forward, Left, Right, Back
            will be held down until the gesture for stop issued

        Camera
            will be pressed whenever the gesture is made
            continuously holding up the gesture will only press the binding once
        
        Sprint
            will be held as long as the gesture is held up
        
        Stop
            stops all inputs

    Parameters:
        gesture: name of hand symbol performed by user
        hand: indicates hand performed the gesture, should be either 'left' or 'right'

    Returns:
        none
    '''
    
    action = gestureToAction(gesture, hand)
    key = bindings[action]

    if action in ['Roll','Attack','Heal']: # mash these actions as long as action is performed
        print(gesture, " ",action, " ", key)
        keyboard.press(key)
        time.sleep(.1)
        keyboard.release(key)
        pass
    if action != prevAction:
        # change only occurs if this is a 'new' input
        print(gesture, " ",action, " ", key)
        match action:
            # deal with new action
            case 'Stop':
                releaseAll()
            case 'Forward'|'Left'|'Right'|'Back'|'Sprint':
                # hold
                print("hold")
                keyboard.press(key)
                match key:
                    # QOL for contradictory movement
                    case 'Forward':
                        key.release(bindings["Back"])
                    case 'Back':
                        key.release(bindings["Foward"])
                    case 'Left':
                        key.release(bindings["Right"])
                    case 'Right':
                        key.release(bindings["Left"])
            case 'Camera':
                print("Tap")
                # Tap
                keyboard.press(key)
                time.sleep(.1)
                keyboard.release(key)
            case _:
                print(action,"not found")

        match prevAction:
            # deal with old action
            case 'Sprint':
                keyboard.release(bindings['Sprint'])

        prevAction = action
                
def releaseAll():
    '''
    Releases any inputs currently being sent to the game

    Parameters:
        none

    Returns:
        none
    '''
    for k in list(bindings.values()):
        if k != "stop":
            keyboard.release(k)

def gestureToAction(gesture:str, hand:str = 'right')->str:
    '''
    Returns the action that corresponds to the given gesture

    Parameters:
        gesture: string indicating what gesture the user performed
        hand: indicates hand performed the gesture, should be either 'left' or 'right'

    Returns:
        name of the action that corresponds with gesture
    '''

    match gesture:
        case 'a'|'e'|'m'|'n'|'s'|'t'|'x':
            gesture = 'fist'
        case 'd'|'l':
            gesture ='one'
        case 'k'|'r'|'u'|'v':
            gesture ='two'
        case 'c'|'o':
            gesture ='o'
        case 'i'|'y':
            gesture = 'pinky'
        case 'q':
            gesture = 'q'
        case _:
            gesture = "Unknown"

    if hand == "right":
        match gesture:
            case 'q': 'Camera'
            case 'fist': 'Attack' 
            case 'o': 'Roll'
            case 'pinky': 'Sprint'
            case 'one': 'Heal'
    else:
        match gesture:
            case 'fist': 'Forward'
            case 'pinky': 'Left'
            case 'o': 'Back'
            case 'one': 'Right'
            case 'q': 'Stop'