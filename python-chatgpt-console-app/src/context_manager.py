class ContextManager:
    def __init__(self):
        self.history = []  # List of dicts: {"role": ..., "content": ...}

    def add_to_history(self, message):
        # message should be a dict: {"role": ..., "content": ...}
        self.history.append(message)

    def get_history(self):
        return self.history
