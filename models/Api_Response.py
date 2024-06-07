from flask import jsonify

class APIResponse:
    @staticmethod
    def success(data=None, message="Success", status_code=200):
        response = {
            "status": "success",
            "message": message,
            "data": data
        }
        return response
        # return jsonify(response), status_code

    @staticmethod
    def failure(message="Failure", status_code=500):
        response = {
            "status": "failure",
            "message": message
        }
        return response
        # return jsonify(response), status_code