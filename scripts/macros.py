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
prevActions:list=[] # last action

def set_zones(zones:int = 1)-> None:
    '''
    Tells script how many camera zones there are

    Parameters:
        zones: number of camera zones there are

    Returns:
        none
    '''

    for i in range(zones):
        prevActions.append('Unknown')

def input(gesture:str, zone:int = 0)->None:
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

    action = gestureToAction(gesture, zone, 'Unknown')
    key = bindings.get(action,'Unknown')
    prevAction = prevActions[zone]

    if action in ['Roll','Attack','Heal']: # mash these actions as long as action is performed
        keyboard.press(key)
        keyboard.release(key)
        pass
    if action != prevAction:
        # change only occurs if this is a 'new' input
        '''
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
        '''
        
        if action != "Unknown":
            keyboard.press(key)
            if action == "Camera":
                keyboard.release(key)
        if prevAction != "Unknown":
            keyboard.release(bindings[prevAction])

        prevActions[zone] = action
                
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

def gestureToAction(gesture:str, zone:int = 0, default:str='Unknown')->str:
    '''
    Returns the action that corresponds to the given gesture

    Parameters:
        gesture: string indicating what gesture the user performed
        zone: indicates which region of the webcam the hand gesture was performed in

    Returns:
        name of the action that corresponds with gesture
    '''

    match zone:
        case 1: # right
            match gesture:
                case 'A':
                    return 'Attack'
                case 'O':
                    return 'Camera'
                case 'G' | 'H':
                    return 'Roll'
                case 'Y'|'F':
                    return 'Heal'
                case _:
                    return default
        case 0: # left
            match gesture:
                case 'A':
                    return 'Forward'
                case 'O':
                    return 'Back'
                case 'G'|'H':
                    return 'Right'
                case 'Y'|'F':
                    return 'Left'
                case _:
                    return default
        case _:
            return default