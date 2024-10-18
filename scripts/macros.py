import time
import keyboard

class ERMacros:
    '''
    handles macros for me to play elden ring
    '''
    def __init__(self):
        # key = hand gesture, value = corresponding action
        self.actions:dict={
            'a': 'Forward',
            'j': 'Left',
            'k': 'Back',
            'l': 'Right',
            'u': 'Camera',
            'h': 'Attack', 
            'o': 'Roll',
            'm': 'Sprint',
            'y': 'Heal',
            'p': 'Stop'
        }
        # key = action, value = button that action is bound to
        self.bindings:dict={
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
        self.lockOnCD = 0
        self.prevAction = "Stop" # last action
        pass

    def __del__(self):
        self.releaseAll()

    def input(self, gesture:str, hand:str = 'right')->None:
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
        
        action = self.gestureToAction(gesture)
        key = self.bindings[action]

        if action in ['Roll','Attack','Heal']: # mash these actions as long as action is performed
            print(gesture, " ",action, " ", key)
            keyboard.press(key)
            time.sleep(.1)
            keyboard.release(key)
            pass
        if action != self.prevAction:
            # change only occurs if this is a 'new' input
            print(gesture, " ",action, " ", key)
            match action:
                # deal with new action
                case 'Stop':
                    self.releaseAll()
                case 'Forward'|'Left'|'Right'|'Back'|'Sprint':
                    # hold
                    print("hold")
                    keyboard.press(key)
                    match key:
                        # QOL for contradictory movement
                        case 'Forward':
                            key.release(self.binding["Back"])
                        case 'Back':
                            key.release(self.binding["Foward"])
                        case 'Left':
                            key.release(self.binding["Right"])
                        case 'Right':
                            key.release(self.binding["Left"])
                case 'Camera':
                    print("Tap")
                    # Tap
                    keyboard.press(key)
                    time.sleep(.1)
                    keyboard.release(key)
                case _:
                    print(action,"not found")

            match self.prevAction:
                # deal with old action
                case 'Sprint':
                    keyboard.release(self.bindings['Sprint'])

            self.prevAction = action
                    
    def releaseAll(self):
        '''
        Releases any inputs currently being sent to the game

        Parameters:
            none

        Returns:
            none
        '''
        for k in list(self.bindings.values()):
            if k != "stop":
                keyboard.release(k)

    def getureToAction(gesture:str, hand:str = 'right')->str:
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