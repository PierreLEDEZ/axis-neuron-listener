class Action:
    def __init__(self, name):
        self.name = name
        self.done = False

    def get_name(self):
        return self.name
    
    def is_done(self):
        return self.done

    def set_done(self, done):
        self.done = done

    def __str__(self):
        res = ""
        if self.done: res += "\033[92m"
        res += self.name
        if self.done: res += "\033[0m"

        return res