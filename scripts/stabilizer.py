class Stabilizer:
    """
    Class to stabilize data and remove hickups from the model.
    """

    required : int
    default : str
    confidence_threshold : float

    # Sequential Stabilize Variables
    concurrent_inputs : int = 0
    current_input : str

    # Group Stabilize Variables
    index : int = 0
    stored_inputs : list

    def __init__(self, default : str = "Unknown", required : int = 3, confidence_threshold : float = 0.8):
        """
        Parameters:
            default (str) : The default input to return when a streak is broken.
            Defaults to "Unknown".
            required (int) : The number of consecutive times a gesture must be
            detected. Defaults to 3.
            confidence_threshold (float) : The required confidence for an input
            to be valid. Defaults to 0.8.
            
        Returns:
            none
        """
        self.default = default
        self.required = required
        self.current_input = self.default
        self.confidence_threshold = confidence_threshold
        self.stored_inputs = [default] * ((required * 2) - 1)

    def sequential_stabilize(self, input_label : str, confidence : float) -> str:
        """
        Checks if input has shown up a certain number of times in a row.
        If confidence check fails, input is treated as the default value.

        Parameters:
            input_label (str) : Input to compare.
            confidence (float) : The confidence value of your input.
            
        Returns:
            str : The current input label or default if current input hasn't meet the
            reuired occurences.
        """
        if confidence < self.confidence_threshold:
            input_label = self.default

        if input_label == self.current_input:
            self.concurrent_inputs += 1 
        else:
            self.concurrent_inputs = 1 
            self.current_input = input_label

        if self.concurrent_inputs >= self.required:
            return self.current_input
        else:
            return self.default
        
    
    def group_stabilize(self, input_label : str, confidence : float) -> str:
        """
        Adds input_label to a stored inputs list, replacing the oldest value.
        If confidence check fails, input is treated as the default value.

        Parameters:
            input_label (str) : Input to compare.
            confidence (float) : The confidence value of your input.

        Returns:
            str : The most frequent label in stored_labels.
        """
        if confidence < self.confidence_threshold:
            input_label = self.default

        self.stored_inputs[self.index] = input_label
        
        self.index = (self.index + 1) % len(self.stored_inputs)

        return max(self.stored_inputs, key=self.stored_inputs.count)

        

# TESTING
# TODO : REMOVE AFTER VALIDATION
# def main():
#     stab = Stabilizer(required=3)
#     print(stab.group_stabilize("a", .9))
#     print(stab.group_stabilize("a", .8))
#     print(stab.group_stabilize("b", .8))
#     print(stab.group_stabilize("b", .5))
#     print(stab.group_stabilize("a", .9))
#     print(stab.group_stabilize("b", .9))
#     print(stab.group_stabilize("b", .9))
#     print(stab.group_stabilize("a", .9))
# main()