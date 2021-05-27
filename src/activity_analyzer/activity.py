from .action import Action

class Activity:
    def __init__(self, name, actions):
        self.name = name
        self.complete = False
        self.create_action_list(actions)
        self.current_action_index = 0
        self.current_action = self.actions[0]

    def create_action_list(self, action_list):
        self.actions = {}
        for action_index in range(len(action_list)):
            action_object = Action(action_list[action_index])
            self.actions[action_index] = action_object

    def go_to_next_action(self):
        self.current_action.set_done(True)
        if (self.current_action_index+1) in self.actions:
            self.current_action_index += 1
            self.current_action = self.actions[self.current_action_index]
        else:
            self.complete = True

    def is_complete(self):
        return self.complete

    def reset(self):
        self.complete = False
        self.current_action_index = 0
        self.current_action = self.actions[self.current_action_index]
        for action in self.actions:
            self.actions[action].set_done(False)

    def __str__(self):
        res = ""
        res += "{}: ".format(self.name)
        for action in self.actions:
            res += self.actions[action].__str__()
            if action < len(self.actions.keys()) - 1:
                res += " -> "
        res += "\n"
        return res