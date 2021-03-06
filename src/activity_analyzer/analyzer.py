import os, time, threading
from .action import Action
from .activity import Activity


class ActivityAnalyzer(threading.Thread):
    
    def __init__(self, activity_list, queue, mode):
        """
        """

        threading.Thread.__init__(self)
        self.create_activity_list(activity_list)
        self.action_queue = queue
        self.mode = mode
        self.killed = False
        self.started = False

    def create_activity_list(self, raw_activity_list):
        """
        """

        self.activities = []
        for raw_activity_name in raw_activity_list:
            activity = Activity(raw_activity_name, raw_activity_list[raw_activity_name])
            self.activities.append(activity)

    def check_activities(self, prediction):
        """
        """

        finished_activity = None
        for activity in self.activities:
            if activity.current_action.name == prediction:
                activity.go_to_next_action()
                if activity.is_complete(): finished_activity = activity
        return finished_activity

    def find_current_activities(self):
        """
        """

        current_length = 0
        current_activities = []
        current_activities_name = []
        for activity in self.activities:
            if activity.current_action_index == current_length:
                current_length = activity.current_action_index
                current_activities.append(activity)
                current_activities_name.append(activity.name)
            elif activity.current_action_index > current_length:
                current_length = activity.current_action_index
                current_activities.clear()
                current_activities.append(activity)
                current_activities_name.clear()
                current_activities_name.append(activity.name)
        return current_activities, current_activities_name

    def resume_activities(self):
        """
        """

        for activity in self.activities:
            print(activity)

    def reset(self):
        """
        """

        for activity in self.activities:
            activity.reset()

    def prediction_results(self, current_activities, current_activities_name, finished_activity):
        """
        """

        if finished_activity != None:
            print("Finished activity: {}".format(finished_activity.name))
            self.reset()
        else:
            print("Current activity / activities: {}".format(" OR ".join(current_activities_name)))
            next_actions = []
            for activity in current_activities:
                next_actions.append(activity.current_action.name)
            print("Possible next actions: {}".format(" OR ".join(next_actions)))

    def clear_screen(self):
        """
        """

        if os.name == "nt": os.system("cls")
        else: os.system("clear")

    def training_mode(self):
        """
        """

        quit_choice = False

        while not self.killed:
            self.started = False
            self.clear_screen()
            print("--- TRAINING mode ---\n\n")
            for index, activity in zip(range(1, len(self.activities)+1), self.activities):
                print("{} - {}".format(index, activity), end="")
            print("\n")
            input_choice = False
            while not input_choice:
                activity_choice = input("Choose one activity:\n")
                if activity_choice.isdigit() and int(activity_choice) < len(self.activities)+1 and int(activity_choice) > 0: input_choice = True
                
            activity_to_train = self.activities[int(activity_choice)-1]

            activity_complete = False
            while not activity_complete:
                self.started = True

                current_prediction = self.action_queue.get()
                self.clear_screen()
                print("--- TRAINING mode ---\n\n")
                print("Current prediction: {}\n\n".format(current_prediction))
                if activity_to_train.current_action.name == current_prediction:
                    activity_to_train.go_to_next_action()
                    if activity_to_train.is_complete(): activity_complete = True
                else:
                    print("Incorrect !")
                print("-> Next action to do: {}".format(activity_to_train.current_action.name))

                print("=================================\n")

                print(activity_to_train)

                if activity_complete:
                    print("Well done, you complete this activity !")
            activity_to_train.reset()
    
    def recognition_mode(self):
        """
        """

        while not self.killed:

            self.started = True
            finished_activity = None

            current_prediction = self.action_queue.get()
            self.clear_screen()
            print("--- RECOGNITION mode ---\n\n")
            print("Current prediction: {}\n\n".format(current_prediction))
            finished_activity = self.check_activities(current_prediction)

            current_activities, current_activities_name = self.find_current_activities()

            self.resume_activities()

            self.prediction_results(current_activities, current_activities_name, finished_activity)


    def run(self):
        """
        """

        if self.mode == "recognition":
            self.recognition_mode()
        else:
            self.training_mode()