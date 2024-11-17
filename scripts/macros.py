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
    'Jump': 'space',

    'Camera': 'q', 
    'Heal': 'f',
    'Interact':'e',

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
    default = "Unknown"
    action = gestureToAction(gesture, zone, default)
    key = bindings.get(action,default)
    prevAction = prevActions[zone]

    if action in ['Roll','Attack','Heal','Interact']: # mash these actions as long as action is performed
        keyboard.press(key)
        time.sleep(.05)
        keyboard.release(key)
        pass
    if action != prevAction:
        # change only occurs if this is a 'new' input
        if action != default:
            keyboard.press(key)
        if prevAction != default:
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
    Returns the action that corresponds to the given gesture and zone

    Parameters:
        gesture: string indicating what gesture the user performed
        zone: indicates which region of the webcam the hand gesture was performed in

    Returns:
        name of the action that corresponds with gesture
    '''
    conversions = [
        { 
            'Forward': ['A','N'],
            'Back': ['O','C','P'],
            'Left': ['Y','F'],
            'Right': ['L','T','V','X'],
            'Interact': ['G','H']
        },
        { 
            'Attack': ['A','N'],
            'Heal': ['Y','F'],
            'Camera': ['O','C','P'],
            'Roll': ['L','T','V','X'],
            'Jump' : ['G','H']
        }
    ]

    conversion = conversions[zone]

    for action in conversion.keys():
        if gesture in conversion[action]:
            return action
    return default