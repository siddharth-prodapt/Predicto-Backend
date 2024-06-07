from enum import Enum

class HttpMethod(Enum):
    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'


class APIEndpoint:
    """
    Represents an API endpoint configuration.

    Attributes:
    - id (int): The unique identifier for the API endpoint.
    - endpoint (str): The endpoint path (e.g., '/api/v1/users').
    - name (str) : Main functionlity of endpoint
    - desc (str) : Description of endpoint
    - method (HttpMethod): The HTTP method used by the endpoint (GET, POST, PUT, DELETE).
    - params (list of APIParam): The parameters expected by the endpoint.
    """
    def __init__(self, id, name,desc, endpoint, method, params=[]):
        self.id = id
        self.name = name
        self.desc = desc
        self.endpoint = endpoint
        self.method = method
        self.params = params

    def to_dict(self):
        return {
            "id": self.id,
            "endpoint": self.endpoint,
            "method": self.method,
            "desc" : self.desc,
            "params": [param.to_dict() for param in self.params]
        }

class APIParam:
    """
    Represents a parameter of an API endpoint.

    Attributes:
    - in_ (str): The parameter location (e.g., 'formData').
    - name (str): The parameter name.
    - required (bool): Indicates whether the parameter is required.
    - type_ (str): The parameter type.
    - description (str): A description of the parameter.
    """
    def __init__(self, in_:str, name:str, required:bool, type_:str, description:str, defaultValue: any):
        self.in_ = in_
        self.name = name
        self.required = required
        self.type_ = type_
        self.description = description
        self.defaultValue = defaultValue

    def to_dict(self):
        """
        Converts the APIParam instance to a dictionary.

        Returns:
        - dict: A dictionary representation of the APIParam instance.
        """
        return {
            "in": self.in_,
            "name": self.name,
            "required": self.required,
            "type": self.type_,
            "description": self.description,
            "defaultValue":self.defaultValue
        }

# Usage
# params = [
#     APIParam("formData", "customer_email", True, "string", "Customer Email")
# ]

# api_endpoint = APIEndpoint(1, "/api/v1/users", "POST", params)
# print(api_endpoint.to_dict())
