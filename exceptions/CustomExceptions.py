class InvalidRequest(Exception):
    def __init__(self, message="Invalid Request"):
        super().__init__(message)
