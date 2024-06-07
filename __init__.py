from flask import Flask, request, jsonify,send_from_directory, make_response, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.attributes import flag_modified
from dataclasses import dataclass
import os
from sqlalchemy import func, CheckConstraint
import json
from test import load_data_from_url
import numpy as np
from models import  DataSetup, PreProcessing, ML_Models, PCA as Pca, Kmeans
from models.Api_Response import APIResponse
# from flask_restful import Resource, Api
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from exceptions import CustomExceptions
from queue import Queue
from models.API_Handler import APIEndpoint, APIParam, HttpMethod
from models import Visualization
import base64

from dotenv import load_dotenv
load_dotenv() # take environment variables from .env.

# init configurations
app = Flask(__name__)
api = Api(app, title='PREDICTO', description='Empowering Your Predictions with Precision and Ease', doc="/swagger-ui")
CORS(app)
db = SQLAlchemy()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)

DATA_FRAME = None
DATA_FRAME_DB = []
MODEL = None
X = None
Y = None
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = None, None, None, None
API_HANDLER_LIST = []
HISTORY_LIST = []
EXECUTION_QUEUE = Queue()

USER_FILE_DS = {
    
}

class DataFrame_DS:
    def __init__(self , id, dataframe):
        self.id = id
        self.dataframe = dataframe
        
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataframe": self.dataframe
        }

        


upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                            type=FileStorage, required=True)

filename_parser = api.parser()
filename_parser.add_argument('filename',
                            type=str, required=True, help= 'Filename to get data')

filepath_parser = api.parser()
filepath_parser.add_argument('filepath',
                            type=str, required=True, help= 'Absolute file path to download the file')

limit_parser = api.parser()
limit_parser.add_argument('limit', type=int, required=False, help='Nos of data required. Default=5')

columnName_parser = api.parser()
columnName_parser.add_argument('column_name', type=str, required=True)

column_index_parser = api.parser()
column_index_parser.add_argument('columnIndex1', type=int, required=True)
column_index_parser.add_argument('columnIndex2', type=int, required=True)

drag_column_parser = api.parser()
drag_column_parser.add_argument('fromIndex', type=int, required=True)
drag_column_parser.add_argument('toIndex', type=int, required=True)

id_filename_parser = api.parser()
id_filename_parser.add_argument('id', type=int, required=True)
id_filename_parser.add_argument('filename', type = str, required=True)

calculated_column_parser = api.parser()
calculated_column_parser.add_argument('newColumnName', type=str, required=True)
calculated_column_parser.add_argument('calculations', type=str, required=True)

null_with_stats_parser = api.parser()
null_with_stats_parser.add_argument("column_name", type = str, required = True)
null_with_stats_parser.add_argument("strategy", type = str, required = True, default="mean")

null_with_attr_parser = api.parser()
null_with_attr_parser.add_argument("column_name", type = str, required = True)
null_with_attr_parser.add_argument("attribute", type = str, required = True, default="mean")

unique_element_parser = api.parser()
unique_element_parser.add_argument("column_name", type = str, required = True)

threshold_parser = api.parser()
threshold_parser.add_argument('threshold', type = float, required  = False, default = 0.95)

num_components_parser = api.parser()
num_components_parser.add_argument('num_component', type = int, required = True)

kmeans_clustering_parser = api.parser()
kmeans_clustering_parser.add_argument('nos_of_cluster', type = int, required = True)
# Models
class Users(db.Model):
    __tablename__ = "Users"
    id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100),nullable=False)
    create_at = db.Column(db.DateTime(timezone=True), server_default= func.now())
    files = db.Column(JSONB)

    def __repr__(self):
        return f'id: {self.id} , firstName: {self.firstName},email: {self.email}, create_at: {self.create_at}, files: {self.files}'

    def to_dict(self):
        return {'id': self.id , 'firstName': self.firstName,'email': self.email, 'create_at': str(self.create_at), 'files': self.files}

class SupervisedLearningAlgorithms(db.Model):
    __tablename__ = "SL_algorithms"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(100), nullable=False, unique = True)
    category = db.Column(db.String(100), nullable=False)
    
    __table_args__ = (
        CheckConstraint(
            category.in_(['classification', 'regression']),
            name='valid_category'
        ),
    )
    
    def __repr__(self):
        return f"id={self.id}, name={self.name}, value={self.value}"

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'value': self.value
        }
    

signup_dao = api.model('Signup', {
    'firstname': fields.String(required = True, description = "Firstname of user"), 
    'email': fields.String(required=True, description="User Email Id"),
    'password': fields.String(requrired=True, description="User password")
})

signin_dao = api.model('Signin', {
    'email': fields.String(required=True, description="User Email Id"),
    'password': fields.String(requrired=True, description="User password")
})

userId_filename_dto = api.model('UserID_Filename',{
    'userId': fields.Integer(required=True, description="User Email Id", example = 25),
    'filename': fields.String(required=True, description="Filename to get data", example = "ML-MATT-CompetitionQT1920_test.csv")
})

split_data_body = api.model('Split Data', {
    'target_columns' : fields.List(fields.String, required = True, example= ['col1', 'col2', 'col3'])
})

train_test_split_data = api.model('Split Data', {
    'target_columns' : fields.List(fields.String, required = True, example= ['col1', 'col2', 'col3'], description = 'list of str, the names of the target columns'),
    'test_size': fields.Float(required = False, default = 0.2, description = 'the proportion of the dataset to include in the test split'),
    'random_state': fields.Integer(required=False, description = 'controls the random seed for reproducibility', default = None )
}) 

create_model_dto = api.model('Create Model', {
    'model_type': fields.String(required = True, 
                                type = str, 
                                help = 'model_type cannot be blank',
                                description="A string specifying the type of machine learning model", 
                                example="linear_regression", 
                                enum =['linear_regression', 'logistic_regression',
                                    'decision_tree_classifier', 'decision_tree_regressor',
                                    'random_forest_classifier', 'random_forest_regressor',
                                    'svc', 'svr', 'kneighbors_classifier', 'kneighbors_regressor',
                                    'gradient_boosting_classifier', 'gradient_boosting_regressor',
                                    'gaussian_nb', 'mlp_classifier', 'mlp_regressor'])
})
figSize_dto = api.model('figsize', {'width':fields.Integer(requried = False, example = 10),
                                    'height': fields.Integer(required = False, example = 6)})

boxplot_body_dto = api.model('Boxplot Body', {
    'userId': fields.Integer(required=True, description="User Id", example=25),
    'x_label': fields.String(required=False, type=str, description="X Axis label for Box Plot", example='X-axis label'),
    'y_label': fields.String(required=False, type=str, description="Y Axis label for Box Plot", example='Y-axis label'),
    'xticks_rotation': fields.Integer(required=False, description="Rotation angle for the x-axis labels in degrees. Default is 45", example=45),
    'figsize': fields.Nested(figSize_dto)
})
file_rename_body = api.model('Rename File ', {
    'id': fields.Integer(required = False, type = int, description = "Id of the file"),
    'orginal_filename': fields.String(required = True, type = str, description = "Original Filename"),
    'new_filename'    : fields.String(required = True, type = str, description = "Original Filename")
})

supervised_learning_algo_dto = api.model('Supervised Learning Algo', {
    'name' : fields.String(required = True, description = 'The Representation name of Algorithm'),
    'value': fields.String(required = True, description = 'Value of algo text'),
    'category' : fields.String(required = True, description = 'category of algorithm', enum =['classification', 'regression'])
})

plot_correlation_matrix_dto = api.model('Plot Correlation Matrix', {
    'method': fields.String(required = False, description = 'Method used to calculate the correlation. Default is \'pearson\'',
                            enum = ['kendall', 'spearman', 'pearson'], example = 'pearson'),
    'annot': fields.Boolean(required = False, 
                            description = 'Whether to annotate the correlation values on the plot. Default is True ',
                            example = True),
    'cmap' : fields.String(required = False, description = 'The colormap to be used for the plot. Default is coolwarm',
                           example = 'coolwarm'),
    'figsize': fields.Nested(figSize_dto),
})

barchart_dto = api.model('Barchart', {
    'categories' : fields.List(required = True, 
                               type = str,
                               description = 'List of categories or labels for the x-axis',
                               cls_or_instance= fields.String
                               ),
    'values' : fields.List(required = True, 
                           type = any, 
                           description = 'List of values corresponding to each category',
                           cls_or_instance = fields.String
                           ),
    'title': fields.String(required = True, 
                           description = 'Title of the Chart', 
                           example = 'Title'),
    'xlabel' : fields.String(required = True,
                             description = 'Label for the x-axis'),
    'ylabel' : fields.String(required = True, 
                             description = 'Label for the y-axis'),      
})

barchart_df_dto = api.model('Barchart form df', {
    'x_columns' : fields.List(required = True, cls_or_instance= fields.String,
                              description = 'List of columns for the x-axis',
                              ),
    'y_columns' : fields.List(required = True, cls_or_instance= fields.String,
                              description = 'List of columns for the y-axis',
                              ),
    'title' : fields.String(required = True, 
                            description = 'Title of chart'),
    'xlabel': fields.String(required = True, 
                            description = 'Label for the X-axis'),
    'ylabel': fields.String(required = True, 
                            description = 'Label for the y-axis')
})

cus_scatter_plt_dto = api.model('Custom Scatter Plot', {
    'x_col' : fields.String(requrired = True, 
                            description = 'The column name for the x-axis.'),
    
    'y_col' : fields.String(requrired = True, 
                            description = 'The column name for the x-axis.'),
    
    'hue_col' : fields.String(required = False, 
                              description = "The column name for the hue (color grouping)",
                              ),
    'xlabel': fields.String(required = True, 
                            description = 'Label for the X-axis',
                            default = 'Default x-label'),
    'ylabel': fields.String(required = True, 
                            description = 'Label for the y-axis',
                            default = 'Default y-label'),
    'title': fields.String(requried = False,
                          description = 'Title for Scatter Plot',
                          default = 'Default Title')
})

scatterplot_col_dto = api.model('Scatterplot each column', {
    'target_column' : fields.String(required = True, 
                                    description = 'The column name for the target axis (y-axis)'),
    'hue_column' : fields.String(required = False, 
                                 description = 'The column name for the hue (color grouping)'),
    'hue_order' : fields.List(required = False, cls_or_instance= fields.String,
                              description = 'Order for the hue values'),
})

plot_df_dto = api.model('Plot Dataframe Desc', {
    'figsize': fields.Nested(figSize_dto),
    'bar_width' : fields.Float(required = False, 
                               description = 'Width of the bars in the bar chart. Default is 0.35',
                               example = 0.35),
    'error_bars': fields.Boolean(required = False,
                                 description='Boolean indicating whether to include error bars representing standard deviation. Default is True.',
                                 example = True),
})

biplot_dto = api.model('Plot Biplot', {
    'labels': fields.List(cls_or_instance=fields.String, 
                          required = False, 
                          description = "Labels for data points (optional)")
})

cross_validation_dto = api.model('Cross Validation',{
    'cross_validation' : fields.Integer(required = False, 
                                        default = 5, 
                                        description = 'Number of folds for cross-validation')
})

class File:
    def __init__(self, id, filename, filetype, userId):
        self.id = id
        self.filename = filename
        self.filetype = filetype
        self.userId = userId

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'filetype': self.filetype,
            'userId': self.userId
        }
    
class PreprocessingMethodCategory(db.Model):
    __tablename__ = "PreprocessingMethodCategory"
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100), nullable = False)

class PreprocessingMethods(db.Model):
    __tablename__ = "PreprocessingMethod"
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100), nullable = False)
    categoryId = db.Column(db.String(100), nullable = False)

# Create table when server starts by referencing the specified models
with app.app_context():
    db.create_all()

# Utility functions
def get_folder_name(user):
    return f"{user.id}_{user.firstName}"

def addIndex(jsonObject, startingIndex):
    if not isinstance(jsonObject, list):
        raise ValueError("Input must be a list of objects")
    
    for i in range(len(jsonObject)):
        jsonObject[i]["id"] = startingIndex + i

    return jsonObject

def verify_arguments(**kwargs):
    """
    Verify whether keyword arguments are null or empty strings.

    Args:
    **kwargs: Keyword arguments where the key is the variable name and the value is the variable value.

    Raises:
    ValueError: If any argument is null or an empty string.

    Returns:
    bool: True if all arguments are not null or empty strings.
    """
    for name, value in kwargs.items():
        if value is None:
            raise ValueError(f"Variable '{name}' is null.")
        elif isinstance(value, str) and not value.strip():
            raise ValueError(f"String variable '{name}' is empty.")
    return True

def userFilenameValidation(body):
    userId = body.get('userId')
    filename = body.get('filename')
    
    if userId is None:
        return APIResponse.failure(message = "userId not provided", status_code = 400)
    if filename is None:
        return APIResponse.failure(message = "filename is not provided", status_code = 400)

    try:
        user = Users.query.filter_by(id=userId).first()
        if user:
            filenames = [file['filename'] for file in user.files]
            if filename in filenames:
                app.logger.info("Match found for filename: %s", filename)
                return True
    except Exception as e:
        app.logger.error("Error occurred during filename validation: %s", e)
    return False

def getUserByUserId(userId)->Users :
    '''Return User object from db if exists'''
    try:
        user: Users = Users.query.filter_by(id = userId).first()
    except Exception as e:
        raise RuntimeError(f'User Not found with id: {userId}')
    else:
        return user

def deletePngFiles(path: str)->int:
    """
    Delete all PNG files in the specified directory.

    Args:
    - path (str): The path to the directory containing PNG files.

    Returns:
    - int: The number of PNG files deleted.
    """
    try:
        files_deleted = 0
        for filename in os.listdir(path):
            if filename.endswith('.png'):
                os.remove(os.path.join(path, filename))
                files_deleted += 1
            
        return files_deleted
    except Exception as e:
        raise RuntimeError('Invalid dir ')

import shutil
def removePlotFolder(folder_path: str)->bool:
    """
    Remove a folder and its contents.

    Args:
    - folder_path (str): The path to the folder to be removed.

    Returns:
    - bool: True if the folder and its contents were successfully removed, False otherwise.
    """
    try:
        if os.path.exists(folder_path):
            # remove folder and its contents
            shutil.rmtree(folder_path)
            print("Remove Scatter Plot Folder ")
            app.logger.debug(f'{folder_path} removed successfully')
            return True
    except Exception as e:
        return False 

def convertToBase64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
            base64_img = base64.b64encode(img_bytes).decode('utf-8')
            
        # Additional parameters
        additional_params = {
            "image": base64_img
        }
        
        return additional_params
        
        # return APIResponse.success(data = additional_params )
    except Exception as e:
        app.logger.warning('Base 24 Conversion : ', str(e))
        additional_params = {
            "image": "Server Issue"
        }
        return additional_params
       
def get_file_type(filename)->str:
    # Split the filename into base name and extension
    base_name, ext = os.path.splitext(filename)
    # Remove leading and trailing whitespaces from the extension
    ext = ext.strip().lower()
    if ext == '.csv':
        return 'csv'
    elif ext == '.xls':
        return 'xls'
    elif ext == '.xlsx':
        return 'xlsx'
    else:
        return 'unknown'     

# swagger namespaces
pre_ns = api.namespace('Preprocessing Method Category',description='Preprocessing Method Category Controller', path="/preprocessing")
df_ns = api.namespace('Dataframe Controller', description = 'Data Frame Controller', path = "/dataframe")
auth_ns = api.namespace('Authentication Controller', description = 'Authentication Controller', path = "/auth")
file_ns = api.namespace('File Handling Controller', description = 'File controller', path="/user/<int:userId>/file")
data_loading_ns = api.namespace('Dataloading Endpoints Controller', description = 'DataLoading Controller', path = "/dl")
copy_df_ns = api.namespace('Dataframe Controller', description = 'Copy data frame', path = "/dataframe/copy")
preprocessing_ns = api.namespace('Preprocessing Controller', description = 'Preprocessing Methods Endpoints', path="/pre")
normalization_ns = api.namespace('Normalizations Controller', description = 'Several Normalisation Techniques', path='/normalization')
ml_model_ns = api.namespace('ML Model', description = 'ML Model', path='/model')
history_tracker_ns = api.namespace('History Tracker Controller', description = 'History', path='/history') 
supervised_learning_types_ns = api.namespace('Supervised Learning Controller', description='Supervised Learning Types', path='/supervised-learnings')
visualisation_ns = api.namespace('Visualisations Controller', description = 'Endpoints for data visualisations', path = '/visualisations')
pca_ns = api.namespace('Principal Component Analysis', description = 'PCA ', path = '/pca')
kmeans_ns = api.namespace('K Means Clustering', description = 'K Means Clustering', path = '/kmeans')
@df_ns.route('')
@df_ns.doc("Create Dataframe Instance")
class DataFrame(Resource): 
    @df_ns.expect(userId_filename_dto)
    @df_ns.doc("Instantiate a Dataframe Instance")
    def post(self):
        '''
        Instantiate a Dataframe Instance
        '''
        app.logger.info("POST: Create data frame instance")
        try:
            body = request.json
        except Exception as e:
            return APIResponse.failure(message = "Request body not provided", status_code=400)
            
        app.logger.debug("Global DataFrame is instantiated")
        
        if not userFilenameValidation(body):
            return APIResponse.failure(message="Invalid Request Body", status_code = 400)

        try:
            userId = body.get('userId')
            filename = body.get('filename')
            
            user = Users.query.filter_by(id=userId).first()
            
            # File path from where to load data in dataframe
            new_file_path = f"http://localhost:8000/user/{user.id}/file?filename={filename}"
            
            df = DataSetup.load_data_from_url(new_file_path)
            
            if df is None:
                return APIResponse.failure(message = "Invalid File Path or df is none")
            
            global DATA_FRAME
            DATA_FRAME = df
            
            # print("\n\n\n\nDATAFRAME CREATED: ", DATA_FRAME)
            
            app.logger.info("Dataframe Created Successfully")
            # return jsonify({"message": "success"})
            
            app.logger.debug('History list is reset')
            API_HANDLER_LIST.clear()
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Instantiate file',
                desc     = 'File has been instatiated for further operations', 
                endpoint = '/dataframe',
                method   = 'POST',
                )
            
            # api_endpoint.params = None
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(status_code=201, message="Dataframe Created Successfully"),201
        
        except Exception as e:
            app.logger.warn("Exception occured", str(e))
            return APIResponse.failure(message = str(e))
        
    def delete(self):
        return APIResponse.success(message = "Dummy Function")

@df_ns.route('/save')
class SaveDataframe(Resource):
    @df_ns.expect(userId_filename_dto)
    def post(self):
        """
    Saves a pandas DataFrame to a specified location with a specified name.
    
    Returns:
        str: The full path of the saved file.
    """
        try:
            body = request.json
            print("Body: ", body)
            
            app.logger.debug(f'Body Received: {body}')
            
            user_id = body.get('userId')
            filename = body.get('filename')
            
            user = Users.query.filter_by(id=user_id).first()
            app.logger.debug(f'User fetched from DB: {user}')
            
            if user is None:
                raise RuntimeError("Invalid user ID ")
            
            
            file_path = f'{UPLOAD_FOLDER}/{get_folder_name(user)}/dataframes_copy'
            
            # if datafarames_copy directory not present then create the directory
            if not os.path.exists(file_path):
                os.mkdir(os.path.join(file_path))
            
            app.logger.info("Dataframe is going to save as csv file")
            app.logger.debug(f'File Path: {file_path}: filename: {filename}')
            
            if DATA_FRAME is None:
                raise RuntimeError("Dataframe not instantiated")
            
            full_file_path = DataSetup.save_dataframe(DATA_FRAME, file_path, filename)

            app.logger.info("Dataframe is saved to file successfully")
            return APIResponse.success(data = {"filepath": full_file_path}, message="File saved successfully at filepath"), 201
        
        except FileNotFoundError as e:
            app.logger.exception("File Not Found Error: ", str(e))
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.exception("Exception occured while saving file: ", e)
            return APIResponse.failure(message = str(e)), 500

@copy_df_ns.route('')
class CreateDataFrameCopy(Resource):
    def post(self):
        try:
            if DATA_FRAME is None:
                return APIResponse.failure(message="Dataframe not instantiated", status_code=400)
            
            df_obj = {
                "id": len(DATA_FRAME_DB)+1,
                "name": str(len(DATA_FRAME_DB)+1)+"_DATA",
                "dataframe": DataSetup.copy_dataframe(DATA_FRAME)
                }
            
            DATA_FRAME_DB.append(df_obj)
            
            if len(DATA_FRAME_DB) > 0 :
                id = DATA_FRAME_DB[len(DATA_FRAME_DB)-1]["id"]
                name = DATA_FRAME_DB[len(DATA_FRAME_DB)-1]["name"]
        except Exception as e:
            app.logger.debug("Exception : ", e)
            return APIResponse.failure(message=str(e)),201
            
        app.logger.debug("Dataframe copid successfully")
        return APIResponse.success(data = {"id": id, "name":name}, message= f"Data frame copied successfully: {id}, {name}"),201

@copy_df_ns.route('/deep-copy')
class CreateDataFrameDeepCopy(Resource):
    def post(self):
        try:
            if DATA_FRAME is None:
                return APIResponse.failure(message="Dataframe not instantiated", status_code=400)
            
            df_obj = {
                "id": len(DATA_FRAME_DB)+1,
                "name": str(len(DATA_FRAME_DB)+1)+"_DATA",
                "dataframe": DataSetup.deep_copy_dataframe(DATA_FRAME)
                }
            
            DATA_FRAME_DB.append(df_obj)
            
            if len(DATA_FRAME_DB) > 0 :
                id = DATA_FRAME_DB[len(DATA_FRAME_DB)-1]["id"]
                name = DATA_FRAME_DB[len(DATA_FRAME_DB)-1]["name"]
        except Exception as e:
            app.logger.debug("Exception : ", e)
            return APIResponse.failure(message=str(e)),201
            
        app.logger.debug("Dataframe copid successfully")
        return APIResponse.success(data = {"id": id, "name":name}, message= f"Data frame deep copied successfully: {id}, {name}"),201

@copy_df_ns.route('/list')
class CopyDataframeList(Resource):
    def get(self):
        """
        Return : Get Copy Dataframe list
        """
        copy_df_list = []
        
        for df_obj in DATA_FRAME_DB:
            id = df_obj["id"]
            name = df_obj["name"]
            
            if id and name:
                df_dto= {"id": id, "name": name}
                copy_df_list.append(df_dto)
            
        return APIResponse.success(data = copy_df_list, message = "Copy Dataframe List")
            

@copy_df_ns.route('/pick/<int:df_id>')
class PickCopyDataframe(Resource):
    def post(self,df_id):
        """
        Select Copied Dataframe using ID
        """
        try:
            list_id = df_id-1

            app.logger.debug("Data_Frame_DB_list[listId] : list_id", list_id)
            print("\n\n\nLIst Id : ", list_id)
            if list_id <= len(DATA_FRAME_DB) and list_id > -1:
                dataframe = DATA_FRAME_DB[list_id]["dataframe"]
            
                print("\n\ndataframe : ", dataframe)
                global DATA_FRAME
                DATA_FRAME = dataframe
                return APIResponse.success(message="New Dataframe Instantiated"), 201
            else:
                return APIResponse.failure(message = "Invalid copy df id"), 400
        except Exception as e:
            app.logger.exception(f'Copy Dataframe By ID encountered Exception: {e}', exc_info=True) 
            app.logger.debug("passed id is not correct: len(copy_data_frame db) - %s and Id: %s", len(DATA_FRAME_DB), list_id)
            return APIResponse.failure(message = "Invalid Request copy file id"), 400
@copy_df_ns.route('/<int:df_id>')
class CopyDataFrameById(Resource):
    def get(self, df_id):
        try:
            if df_id > len(DATA_FRAME_DB) or df_id <len(DATA_FRAME_DB):
                return APIResponse.failure(message= "df_id is invalid!"),400
            
            df_obj = DATA_FRAME_DB[df_id]
            df = df_obj['dataframe']
            
            global DATA_FRAME
            DATA_FRAME = df
            
        except Exception as e:
            app.logger.exception(f'Copy Dataframe By ID encountered Exception: {e}', exc_info=True)
            return APIResponse.failure(str(e)), 500    
        return APIResponse.success(message="Dataframe copied successfully")

        
        
@auth_ns.route('/signup')
class Signup(Resource):
    @auth_ns.expect(signup_dao)
    @auth_ns.doc("Create User")
    def post(self):
        body = request.json
    
        if 'email' not in body or 'password' not in body:
            return APIResponse.failure("Request body does not contain required field", 400)

        email = body.get('email')
        password = body.get('password')
        firstname = body.get('firstname')

        # Check for none
        if email is None or password is None or firstname is None:
            return APIResponse.failure("Invalid Request Body", 400)

        # remove the whitespaces
        email = email.strip()
        password = password.strip()
        firstname = firstname.strip()

        app.logger.debug("Email : %s , Password: %s , Firstname: %s", email, password,firstname)
        # validation check : whether its empty or not
        if len(email) == 0:
            return APIResponse.failure("Email is empty", 400)
        elif len(password) == 0:
            return APIResponse.failure("Password field is empty", 400)
        elif len(firstname) == 0:
            return APIResponse.failure("Firstname field is empty", 400)
        # find user from db by email
        user = Users.query.filter_by(email=email).first()
        if user:
                return APIResponse.failure("Email Id already registered", 400)

        user = Users(
                email = email,
                password = password,
                firstName = firstname,
                files = []
            )
        
        app.logger.debug("User details are valid: ",user )

        try:
            db.session.add(user)
            db.session.commit()
            app.logger.debug("user registered successfully")
        except Exception as e:
            app.logger.warn("Exception occured : ", e)
            return APIResponse.failure("User not created", 500)
        
        try:
            folder_name = get_folder_name(user)
            app.logger.debug("Folder name : ", folder_name)
            
            os.mkdir(os.path.join(UPLOAD_FOLDER,folder_name))  
            app.logger.info("Dir created for user successfully")
        except Exception as e:
            return APIResponse.failure(e, 500)
        
        user_res = {
                "id" : user.id,
                "firstname": user.firstName,
                "email": user.email,
                "created_at": str(user.create_at)
            }
            
        return APIResponse.success(data= user_res, status_code= 201),201

@auth_ns.route('/logout')
class Logout(Resource):
    def post(self):
        try:
            cwd = os.getcwd()
            num_deleted_files = deletePngFiles(path = cwd)
            
            app.logger.debug(f'Num of Deleted Files: {num_deleted_files} ')
            print(num_deleted_files)
            
            folder_path = os.path.join(os.getcwd(), 'scatter_plots')
            print(folder_path)
            if os.path.exists(folder_path):
                if removePlotFolder(folder_path):
                    app.logger.debug('Folder removed successfully')
            
            return APIResponse.success(message = "Logout Successfully"), 200
        
        except OSError as e:
            app.logger.warning('OS Error: ', str(e))
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Error in Logout: ', str(e))
            return APIResponse.failure(message = str(e)), 500 
@auth_ns.route('/signin')
class Signin(Resource):
    @auth_ns.doc("Login with credentials and return user data")
    @auth_ns.expect(signin_dao)
    def post(self):
        body = request.json
        auth = body
    
        app.logger.debug("Request Body : ", auth)

        if not auth or not auth.get('email') or not auth.get("password"):
            app.logger.warn("email or password is not provided")
            return APIResponse.failure("Invalid Credentials", 401)
        
        user:Users  = Users.query.filter_by(email = auth.get("email")).first()

        if not user:
            app.logger.warn("User's email not found in database")
            return APIResponse.failure("User Not Registered", 400)
    
        app.logger.info("User Logged in successfully")

        return APIResponse.success(data= user.to_dict())

@auth_ns.route('/user/<int:id>')
class UserDetails(Resource):
    def get(self, id):
        try:
        # Return User Details
            user:Users = Users.query.filter_by(id = id).first()
        except Exception as e:
            app.logger.error("Error Occured in getting data from DB: ",e)
            return APIResponse.failure(message = str(e)), 500
        else:
            return APIResponse.success(data = user.to_dict())
        
@file_ns.route("")
class FileControllerByFilename(Resource):
    # Upload the file 
    @file_ns.expect(upload_parser)
    @file_ns.doc("Upload file in Users directory and store filename in db")
    def post(self, userId):
        app.logger.debug("User ID : ", userId)
        try:
            file = request.files['file']
            user = Users.query.filter_by(id=userId).first()
            
            folder_name = get_folder_name(user)
            app.logger.debug("Folder Name for this user :", folder_name)

            if file:
                filename = file.filename

                # File destination path 
                path = os.path.join(UPLOAD_FOLDER,folder_name, filename)

                # Updated path as per desktop directory
                # File saved to path
                file.save(path)  

                # File object model to store file data
                new_file = File(len(user.files)+1, filename, file.mimetype,userId)
                
                # user.files already contain atleast single file data then create one copy of json data
                if user.files:
                    files = user.files.copy()
                else:
                    files=[]

                # appending uploaded file in db
                files.append(new_file.to_dict())

                # to avoid mutablity issue 
                user.files = files

                app.logger.debug("User files: ", user.files)

                db.session.commit()

                app.logger.info("User Files commited to database")
                app.logger.debug("User Files after commit: ", user.files)

                return APIResponse.success(data = {
                                                    "path": os.path.join(UPLOAD_FOLDER)+"/"+folder_name+"/"+filename,
                                                    "filename": filename
                                                }, 
                                        ),201
            
            # jsonify({"message": "File uploaded successfully", "path": os.path.join(UPLOAD_FOLDER,folder_name, filename)})
            else:
                return APIResponse.failure(message = "No File Attached", status_code = 400)
        except Exception as e:
            return APIResponse.failure(message = str(e), status_code = 500)

    @file_ns.expect(file_rename_body)
    def put(self, userId):
        """
        Rename File stored in storage
        """
        try:
            body = request.json
            original_filename = body.get('orginal_filename')
            new_filename = body.get('new_filename')
            file_id = body.get('id')

            user = Users.query.filter_by(id=userId).first()

            if not user:
                return APIResponse.failure("User not found with given userId"), 404

            files = user.files.copy()

            for file in files:
                 
                if file["filename"] == original_filename and file["id"] == file_id:
                    app.logger.info("File is going to be renamed")
                    app.logger.debug(f'Original Filename Stored: {file["filename"]}  \nRename Filename : {new_filename}')

                    file_path = f'{UPLOAD_FOLDER}/{get_folder_name(user)}'
                    
                    # Update the filename locally
                    current_filepath = os.path.join(file_path, original_filename)
                    new_filepath = os.path.join(file_path, new_filename)
                    os.rename(current_filepath, new_filepath)
                    
                    # Update the filename in the copy of user.files
                    file["filename"] = new_filename
                    user.files = files
                    
                    # flag the updated object with its attributes
                    flag_modified(user, 'files')
                    
                    # flush the changes to databases
                    db.session.commit()
                   
                    return APIResponse.success(data=user.files, message="File name renamed successfully"), 201

            return APIResponse.failure("File not found with given filename and id"), 404
        except Exception as e:
            app.logger.exception("Exception while renaming the file: ", str(e))
            db.session.rollback()
            return APIResponse.failure(message=str(e)), 500

    @file_ns.expect(id_filename_parser)
    def delete(self, userId):
        """
        Delete file by filename/fileId 
        """
        try:
            filename = request.args.get('filename')
            file_id = request.args.get('id')

            user = Users.query.filter_by(id=userId).first()

            if not user:
                return APIResponse.failure("User not found with given userId"), 404

            files = user.files.copy()

            for file in files:
                print(file["filename"], filename, file["id"], file_id,"\n\n")
                print("Comparison: ",file["filename"] == filename, file["id"] == file_id)
                
                if file["filename"] == filename and int(file["id"]) == int(file_id):
                    app.logger.info("File is going to be deleted")
                    app.logger.warning(f'File to be deleted: {file["filename"]}')

                    file_path = f'{UPLOAD_FOLDER}/{get_folder_name(user)}/{file["filename"]}'
                    
                    # Update the filename locally
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
                        del files[int(file_id)-1]
                        user.files = files
                        flag_modified(user, 'files')
                        
                        # flush the changes to databases
                        db.session.commit()
                        return APIResponse.success(message = "File Deleted Successfully"), 200
                    else:
                        print("Filepath: ",file_path)
                        return APIResponse.failure(message = "File not found at path"), 404
            return APIResponse.failure("File does not exist"), 404
        except Exception as e:
            app.logger.exception("Exception while deleting the file: ", str(e))
            db.session.rollback()
            return APIResponse.failure(message=str(e)), 500
        
    @file_ns.expect(filename_parser)
    @file_ns.doc("Download file", params={'userId': 'UserId of registered resource'})
    def get(self, userId):
        try:
            user = Users.query.filter_by(id=userId).first()
       
            folder_name = get_folder_name(user)

            try:
                filename = request.args.get('filename')
            except Exception as e:
                return APIResponse.failure(message = "filename is not present request params", status_code=400)
        
            if  len(filename.strip())==0:
                return APIResponse.failure(message = "filename provided is empty", status_code=400)

            path = app.config['UPLOAD_FOLDER'] + "/"+folder_name+"/"+filename
            file_path = path
      
            if not os.path.exists(file_path):
                return jsonify({"message": "File not found"}), 404
            return send_from_directory(app.config['UPLOAD_FOLDER']+"/"+folder_name,filename, as_attachment=True)
        except Exception as e:
            return jsonify({"message": "Failed to download file", "error": str(e)}), 500
        
@file_ns.route("/list")
class FileControllerList(Resource):
    
    def get(self, userId):
        try:
            user : Users = Users.query.filter_by(id = userId).first()
        except Exception as e:
            app.logger.error("Exception while fetching data from db: ", e)
            return APIResponse.failure(message = "User Not found"), 400
        else:
            file_list = user.files
            return APIResponse.success(data = file_list)

@file_ns.route('/download')
class DownloadFileByFilePath(Resource):
    '''Donwload File by filepath for picle/ model download'''
    @file_ns.expect(filepath_parser)
    def get(self, userId):
        '''Download Pickle file by filepath'''
        try:
            filename = request.args.get(
                'filepath'
            )
            
            return send_file(path_or_file=filename)
        except FileNotFoundError as e:
            app.logger.warning('Download File by filename : ', str(e))
            return APIResponse.failure(), 404
        except Exception as e:
            app.logger.warning(str(e))
            return APIResponse.failure(message = str(e)), 500
@data_loading_ns.doc("Return Head Data")
@data_loading_ns.route("/headData")
class HeadData(Resource):
    @data_loading_ns.expect(limit_parser, id_filename_parser)
    @data_loading_ns.doc("Get Head Data of Instantiated Dataframe")
    
    def get(self):
        app.logger.info("GET : head data")
        # Query Params
        limit_param = request.args.get('limit') or 5
        id_param = request.args.get('id')
        filename = request.args.get('filename')

        if (id_param and filename):
            id = int(id_param)
            if limit_param:
                limit = int(limit_param)
        else:
            return APIResponse.failure(message = 'Either of (id, filename) request query parameters not provided'), 400
            
        body = {
            "id": id,
            "filename": filename
        }
    
        global DATA_FRAME

        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe ")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        try:
            if not userFilenameValidation(body):
                return APIResponse.failure(message="Invalid Request Body", status_code = 400)
        
            head = addIndex( DataSetup
                            .get_head(DATA_FRAME,limit), 1)  
            headData = {}
            headData = {
                "head": head
            }
        except Exception as e:
            return APIResponse.failure(message = str(e), status_code=500)
        return APIResponse.success(data = headData, status_code = 200)
   
@data_loading_ns.route("/tailData")
@data_loading_ns.doc("Return Tail Data")
class TailData(Resource):
    @data_loading_ns.expect(limit_parser, userId_filename_dto )
    @data_loading_ns.doc("Get Tail Data of Instantiated Dataframe")
    def get(self):
        app.logger.info("GET : Tail data")
        # Query Params
        limit_param = request.args.get('limit') or 5
        id_param = request.args.get('id')
        filename = request.args.get('filename')

        if (id_param and filename):
            id = int(id_param)
            if limit_param:
                limit = int(limit_param)
        else:
            return APIResponse.failure(message = 'Either of (id, filename) request query parameters not provided'), 400
            
        body = {
            "id": id,
            "filename": filename
        }

    
        global DATA_FRAME
        
        # print("\n\n\n\nDATAFRAME: ", DATA_FRAME)
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe ")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        try:
            if not userFilenameValidation(body):
                return APIResponse.failure(message="Invalid Request Body", status_code = 400)
        
            tail = addIndex( DataSetup
                            .get_tail(DATA_FRAME,limit), 1)  
            tailData = {}
            tailData = {
                "tail": tail
            }
        except Exception as e:
            return APIResponse.failure(message = str(e), status_code=500)
        return APIResponse.success(data = tailData, status_code = 200)


@data_loading_ns.route("/getDataFrameShape")
@data_loading_ns.doc("Return Frame Shape")
class DataFrameShape(Resource):
    def get(self):
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        jsonData = DataSetup.get_dataframe_shape(DATA_FRAME)
        return APIResponse.success(data = jsonData) 

@data_loading_ns.route("/getDataFrameSize")
@data_loading_ns.doc("Return Frame Size")
class DataFrameSize(Resource):
    def get(self):
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        jsonData = DataSetup.get_dataframe_size(DATA_FRAME)
        return APIResponse.success(data = jsonData) 
  
@data_loading_ns.route("/getTotalNullValues")
@data_loading_ns.doc("Get Total Null Values")
class DataFrameNullValues(Resource):
    def get(self):
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        
        print("\n\nDataframe : ", DATA_FRAME,"\n\n\n")
        total_null_value_count = DataSetup.get_total_null_values(DATA_FRAME)
        
        return APIResponse.success(data = json.loads(total_null_value_count)) 

@data_loading_ns.route("/columnOrder")
@data_loading_ns.doc("Return column Order")
class ColumnnameInOrder(Resource):
    def get(self):
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        jsonData = DataSetup.get_column_names_in_order(DATA_FRAME)
        return APIResponse.success(data = jsonData) 

@data_loading_ns.route("/describeTheData")
@data_loading_ns.doc("Generate descriptive statistics for all columns of the DataFrame, including non-numeric columns")
class DescribeTheData(Resource):
    def get(self):
        '''Generate descriptive statistics for all columns of the DataFrame, including non-numeric columns'''
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
    
        resData = DataSetup.describe_the_data(DATA_FRAME)
        return resData 


@data_loading_ns.route("/describeData")
@data_loading_ns.doc("Descibe Data")
class DescData(Resource):
    def get(self):
        '''Generate descriptive statistics for numeric columns of the DataFrame.'''
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
    
        resData = DataSetup.describe_dataframe(DATA_FRAME)
        return resData 



@data_loading_ns.route("/getDataFrameInfo")
@data_loading_ns.doc("Return Data Frame Info")
class DataFrameInfo(Resource):
    def get(self):
        print("++++++++++++++++ Start of info()")
        app.logger.info("Into Data Frame Info")
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        jsonRes = DataSetup.get_dataframe_info(DATA_FRAME)
        print("\n\n\n\n\n\nREs: ", jsonRes)
        app.logger.info("Exiting from Data from Info")
        print("Exiting...")
        return jsonRes

@data_loading_ns.route("/getDataTypes")
@data_loading_ns.doc("Data Frame Data types")
class DataframeDatatypes(Resource):
    @data_loading_ns.expect(id_filename_parser)
    def get(self):
        app.logger.info("GET : get data types")  
        
        id_param = request.args.get('id')
        filename = request.args.get('filename')

        if (id_param and filename):
            id = int(id_param)
        else:
            return APIResponse.failure(message = 'Either of (id, filename) request query parameters not provided'), 400
            
        body = {
            "id": id,
            "filename": filename
        }
        
        if DATA_FRAME is None:
            app.logger.warn("Global Dataframe not instantiated or is None")
            return APIResponse.failure(message="Data Frame is not instantiated", status_code=400)
        
        if not userFilenameValidation(body):
            return APIResponse.failure(message="Invalid Request Body", status_code = 400)
        
        jsonData = DataSetup.get_data_types(DATA_FRAME)
            
        return jsonData

@data_loading_ns.route("/column")
@data_loading_ns.doc("Delete Column From Data Frame")
class DeleteColumn(Resource):
    @data_loading_ns.expect(columnName_parser)
    def delete(self):
        try:
            columnName = request.args.get('column_name')
            app.logger.info("Column Name : ", columnName)
            
            global DATA_FRAME
            if DATA_FRAME is None:
                app.logger.warn("Global Dataframe not instantiated or is None")
                return APIResponse.failure(message="Data Frame is not instantiated", status_code=400),400

            DATA_FRAME = DataSetup.drop_column(DATA_FRAME, columnName)
            
        except Exception:
            return APIResponse.failure(message = "Invalid request params"), 400
        else:
            # Store endpoint details to track for history
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Delete Column',
                desc     = f'Deleted the selected coloumn : {columnName}',
                endpoint = '/dl/column',
                method   = 'DELETE'
                )
            api_param_list = []
            api_param = APIParam("query", 'column_name', True, 'string', 'Coloumn name of the table', columnName )
            api_param_list.append(api_param)
            api_endpoint.params = api_param_list
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()   
 
@data_loading_ns.route("/interchangeColumn")
@data_loading_ns.doc("Interchange Column From Data Frame")
class InterchangeColumn(Resource):
    @data_loading_ns.expect(column_index_parser)
    def post(self):
        try:
            index1 = request.args.get('columnIndex1')
            index2 = request.args.get('columnIndex2')

            if (index1 and index2):
                global DATA_FRAME
                DATA_FRAME = DataSetup.interchange_columns(DATA_FRAME, index1, index2)
                
                api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Interchange Column',
                endpoint = '/dl/interchangeColumn',
                method   = 'POST',
                desc     = 'Interchange Column Positions'
                )
                api_param_list = []
                
                api_param = APIParam("query", 'columnIndex1', True, 'int', 'Coloumn Index 1' )
                api_param_list.append(api_param)
                
                api_param2 = APIParam("query", 'columnIndex2', True, 'int', 'Coloumn Index 2' )
                api_param_list.append(api_param2)
                
                api_endpoint.params = api_param_list
            
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
                
                return APIResponse.success()
            else:
                return APIResponse.failure(message="Either of params is empty", status_code=400) 
        except Exception as e:
            app.logger.warning("Exception while interchanging column postition: ", str(e))
            return APIResponse.failure(message = str(e)) 
       
@data_loading_ns.route("/dragColumn")
@data_loading_ns.doc("Drag Column From Data Frame as per user")
class DragColumn(Resource):
    @data_loading_ns.expect(drag_column_parser)
    @data_loading_ns.doc("Drag the columns from Data Frame")
    def post(self):
        try:
            fromIndex = int(request.args.get('fromIndex'))
            toIndex = int(request.args.get('toIndex'))

            if (fromIndex and toIndex):
                global DATA_FRAME
                DATA_FRAME = DataSetup.drag_column(DATA_FRAME, fromIndex, toIndex)
                
                api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Drag Column',
                desc     = 'Coloumn Changed ',
                endpoint = '/dragColumn',
                method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="query", name='fromIndex', required=True, type_='integer',description= 'Source Coloumn Index', defaultValue= fromIndex)
                api_param_list.append(api_param)
                
                api_param = APIParam(in_="query", name='toIndex', required=True, type_='integer',description= 'Destination Coloumn Index', defaultValue= fromIndex)
                api_param_list.append(api_param)
                
                api_endpoint.params = api_param_list
                
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
                
                return APIResponse.success()
            else:
                app.logger.debug('Request Params: ', request.args)
                return APIResponse.failure(message = "Either of request params is empty", status_code=400)
        except Exception as e:
            app.logger.exception('Exceptions in Drag Column: ', str(e))
            return APIResponse.failure(message = str(e), status_code=500)


@data_loading_ns.route("/calculateColumn")
@data_loading_ns.doc("Calculate Column From Data Frame")
class CalculateDataframe(Resource):
    @data_loading_ns.expect(calculated_column_parser)
    def post(self):
        body = request.json
        try:
            newColumnName = body.get('newColumnName')
            calculation   = body.get('calculations')
        except Exception as e:
            return APIResponse.failure(message= "Invalid request body", status_code=400)

        if newColumnName and calculation:
            global DATA_FRAME
            DATA_FRAME = DataSetup.add_calculated_column(DATA_FRAME, newColumnName, calculation)

        return APIResponse.success()
        

@data_loading_ns.route("/add-row")
class AddNewRow(Resource):
    def post(self):
        """
        Add a new row to a pandas DataFrame.
        
        Body: 
        new_row_data (dict): A dictionary where keys are column names and values are corresponding values for the new row.
        example: 
        {
            "Col1": "val1",
            "Col2": "val2"
        }
        """
        try:
            body = request.json
            global DATA_FRAME
            DATA_FRAME = DataSetup.add_new_row(df = DATA_FRAME, new_row_data = body)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Add Row',
                desc     = 'A new row added to dataframe',
                endpoint = '/dl/addRow',
                method   = "POST",
            )
            api_param_list = []
            api_param = APIParam("body", 'new_row_data', True, 'json', "Contains new Row data {'col1': 'value'}",request.json )
            api_param_list.append(api_param)
            api_endpoint.params = api_param_list
                
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
                
            return APIResponse.success(message = "New Row Added Successfully"),201
        
        except Exception as e: 
            app.logger.exception("Adding new row: ", str(e))
            return APIResponse.failure(message = str(e)), 500

@data_loading_ns.route("/convert-numeric-col")
class ConvertNumericCol(Resource):
    def post(self):
        """
        Convert columns containing numeric values as strings to numeric data type.
        """
        try:
            global DATA_FRAME
            DATA_FRAME = DataSetup.convert_numeric_columns(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Convert String Columns to Numeric Data type',
                desc     = 'Converted the coloumn containing the numeric data as string data type to numeric',
                endpoint = '/dl/convert-numeric-col',
                method   = 'POST',
                params   = []
            )
                
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(),200
        except Exception as e:
            return APIResponse.failure("Server Issue"), 500
    
@data_loading_ns.route("/convert-column-to-numeric")
class ToNumeric(Resource):
    @data_loading_ns.expect(columnName_parser)
    def post(self):
        ''' Convert a specific column containing numeric values as strings to numeric data type.'''
        column_name = request.args.get('column_name')
        try:
            global DATA_FRAME
            DATA_FRAME = DataSetup.convert_column_to_numeric(DATA_FRAME, column_name)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Convert String Columns to Numeric Data type',
                desc     = 'Convert a specific column containing numeric values as strings to numeric data type.',
                endpoint = '/dl/convert-column-to-numeric',
                method   = 'POST',
                params   = []
            )
            
            api_param = APIParam(in_ = 'query', 
                                 name = 'column_name', 
                                 required=True, 
                                 type_= 'string',
                                 description='Column Name of the Dataframe', 
                                 defaultValue= column_name)
            
            api_endpoint.params.append(api_param)
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message = "Column converted to numeric"), 201
        except Exception as e:
            return APIResponse.failure(message=str(e)), 500
@preprocessing_ns.route('/dropNullValues')
class DropNullValues(Resource):
    def delete(self):
        '''Drop Null Values from Column'''
        global DATA_FRAME
        try:
            DATA_FRAME = PreProcessing.drop_null_values(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Drop Null Values',
                desc     = 'Null value column has been dropped',
                endpoint = '/pre/dropNullValues',
                method   = 'DELETE',
                params   = []
            )
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
                
            return APIResponse.success(message="Null Values dropped Successfully")
    
        except Exception as e:
            app.logger.exception("Drop Null Values : ", e)
            return APIResponse.failure(message = str(e))
        
        
@preprocessing_ns.route('/dropNullValuesColumn')
@preprocessing_ns.doc("Drop Null values")
class DropNullValuesColoumn(Resource):
    @preprocessing_ns.expect(columnName_parser)
    def delete(self):
        '''Drop Null Values from Column'''
        try:
            column_name = request.args.get('column_name')
            
            global DATA_FRAME
            DATA_FRAME = PreProcessing.drop_null_values_column(DATA_FRAME, column_name)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Drop Null values from column',
                desc     = 'Drops rows with null values from a specific column of a DataFrame. ',
                endpoint = '/pre/dropNullValuesColumn',
                method   = 'DELETE',
            )
            api_param_list = []
            api_param = APIParam(in_="query", 
                                 name='column_name', 
                                 required=True, 
                                 type_='string', 
                                 description="Column Name to drop null values", 
                                 defaultValue= column_name )
            api_param_list.append(api_param)
            api_endpoint.params = api_param_list
                
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message="Null Values dropped Successfully")
        except Exception as e:
            app.logger.exception("Drop Null Values column exception : ", e)
            return APIResponse.failure(message = "Server Issue"), 500
        
        


@preprocessing_ns.route('/fillNullWithStats')
@preprocessing_ns.doc("Fill Null with stats")
class FillNullWithStats(Resource):
    @preprocessing_ns.doc("Fill null with stats")
    @preprocessing_ns.expect(null_with_stats_parser)
    def delete(self):
        try:
            column_name = request.args.get('column_name')
            strategy = request.args.get('strategy')
            
            # global DATA_FRAME
            PreProcessing.fill_null_with_stats(DATA_FRAME, column_name, strategy)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Fill Null with Stats',
                desc     = 'Fills null values in a specific column of a DataFrame with mean, median, or mode',
                endpoint = '/pre/fillNullWithStats',
                method   = 'DELETE',
            )
            api_param_list = []
            api_param = APIParam(in_="query", 
                                 name='strategy', 
                                 required=True, 
                                 type_='string', 
                                 description="Strategy can be mean/median/mode", 
                                 defaultValue= strategy )
            api_param_list.append(api_param)
            
            api_param = APIParam(in_          = "query", 
                                 name         = 'column_name', 
                                 required     = True, 
                                 type_        = 'string', 
                                 description  = "Column Name to drop null values", 
                                 defaultValue = column_name)
            api_param_list.append(api_param)
            
            api_endpoint.params = api_param_list
                
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message="Null Values dropped Successfully"),201
        except Exception as e:
            app.logger.exception("Fill Null with stats exception: ", e)
            return APIResponse.failure(message="Server Issue"), 500


@preprocessing_ns.route('/fillNullWithAttr')
class FillNullWithAttr(Resource):
    
    @preprocessing_ns.expect(null_with_attr_parser)
    def delete(self):
        '''
        Fill the null values of a column with a specific attribute chosen by the user.
        '''
        try:
            column_name = request.args.get('column_name')
            attribute = request.args.get('attribute')
    
            PreProcessing.fill_null_with_attribute(DATA_FRAME, column_name, attribute)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Fill Null with attribute',
                desc     = 'Fill the null values of a column with a specific attribute chosen by the user',
                endpoint = '/pre/fillNullWithAttr',
                method   = 'DELETE',
            )
            api_param_list = []
            api_param = APIParam(in_ = "query", 
                                 name = 'column_name', 
                                 required = True, 
                                 type_= 'string', 
                                 description= "Column Name to drop null values", 
                                 defaultValue= column_name )
            api_param_list.append(api_param)
            api_param = APIParam(in_= "query", 
                                 name= 'attribute', 
                                 required= True, 
                                 type_= 'string', 
                                 description= "Attribute can be mean/median/mode", 
                                 defaultValue= attribute )
            api_param_list.append(api_param)
            api_endpoint.params = api_param_list
                
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            
            return APIResponse.success(message = "Dataframe modified"),201
        except Exception as e:
            app.logger.warning("Exception Occured: ", e)
            return APIResponse.failure(message = str(e))



@preprocessing_ns.route('/countUniqueElements')
@preprocessing_ns.doc("Count Unique Elements")
class CountUniqueElements(Resource):
    @preprocessing_ns.doc("Count Unique Elements")
    # @preprocessing_ns.expect(null_with_stats_parser)
    def get(self):
        jsonRes = PreProcessing.count_unique_elements(DATA_FRAME)
        return APIResponse.success(data = jsonRes)

@preprocessing_ns.route('/uniqueElements')
@preprocessing_ns.doc("Get Unique Elements")
class UniqueElements(Resource):
    @preprocessing_ns.doc("Get Unique Elements")
    @preprocessing_ns.expect(unique_element_parser)
    def get(self):
        column_name = request.args.get('column_name')
        global COLUMN_NAME
        COLUMN_NAME = column_name
        
        jsonRes = PreProcessing.get_unique_elements(DATA_FRAME, column_name)
        dictionaryRes = json.loads(json.dumps(jsonRes))
        return APIResponse.success(data = dictionaryRes)

@preprocessing_ns.route('/duplicateRows')
@preprocessing_ns.doc("Delete Duplicate Rows")
class UniqueElements(Resource):
    @preprocessing_ns.doc("Delete Duplicate Rows")
    def delete(self):
        # return deleted df
        try:
            jsonResdf = PreProcessing.delete_duplicate_rows(DATA_FRAME)
        except Exception as e:
            return APIResponse.failure(message = str(e)), 500
        return APIResponse.success(message = "Duplicated Rows has been deleted")

@preprocessing_ns.route('/identify/datetime')
@preprocessing_ns.doc("Identify datetime columns and their subsets (date, time, datetime) in a DataFrame")
class IdentifyDateTimeColumns(Resource):
    @preprocessing_ns.doc("Identify datetime columns and their subsets (date, time, datetime) in a DataFrame.")
    # @preprocessing_ns.expect(null_with_stats_parser)
    def get(self):
        datetime_columns = PreProcessing.identify_datetime_columns(DATA_FRAME)
        return APIResponse.success(data = datetime_columns)


@normalization_ns.route('/z-score-normalisation')
class Z_Score_Normalisation(Resource):
    def post(self):
        '''
        Perform z-score normalization on the numeric columns of input dataframe
        '''
        try:
            global DATA_FRAME
            DATA_FRAME = PreProcessing.z_score_normalization(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Z Score Normalisation',
                desc     = 'Perform z-score normalization on the numeric columns of input dataframe ',
                endpoint = '/normalisation/z-score-normalisation',
                method   = "POST",
                params   = []
            )
                 
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()
        except Exception as e:
            app.logger.error("Exception : ", e)
            return APIResponse.failure(str(e))
        
@normalization_ns.route('/z-score-normalisation-column')
@normalization_ns.doc('Perform z-score normalization on a specific columns of input dataframe')
class Z_Score_Normalisation_Column(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''
        Perform z-score normalization on a specific columns of input dataframe
        '''
        try: 
            column_name = request.args.get('column_name')
            if column_name:
                global DATA_FRAME
                DATA_FRAME = PreProcessing.z_score_normalization_column(DATA_FRAME, column_name)
                
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Z-Score-Normalisation-Column',
                    desc     = 'Performed z-score normalization on a specific selected columns',
                    endpoint = '/normalization/z-score-normalisation-column',
                    method   = 'POST',
                )
                api_param_list = []
                api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column Name to select for Z-score normalisation", 
                                    defaultValue= column_name )
                api_param_list.append(api_param)
                
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
                
                return APIResponse.success()
            else:
                return APIResponse.failure(message = "column_name is not provided")
        except ValueError as e:
            app.logger.error("Value Error: ",e)
            return APIResponse.failure(message=str(e))
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message=str(e)),500
            
        


@normalization_ns.route('/mean-normalisation')
@normalization_ns.doc('Perform mean normalization on the numeric columns')
class Mean_Normalisation(Resource):
    
    def post(self):
        '''
        Perform mean normalization on the numeric columns
        '''
        try: 
            global DATA_FRAME
            DATA_FRAME = PreProcessing.mean_normalization_dataframe(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Mean Normalisation',
                desc     = 'Performed mean normalisation on the numeric columns',
                endpoint = '/normalization/z-score-normalisation-column',
                method   = 'POST',
            )
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
                
            
            return APIResponse.success()
        
        except ValueError as e:
            app.logger.error("Value Error: ", e)
            return APIResponse.failure(message=str(e))
        except Exception as e:
            app.logger.error("Error: ", e)
            return APIResponse.failure(message=str(e)),500
            


@normalization_ns.route('/mean-normalisation-column')
@normalization_ns.doc('Perform mean normalization on a specific column')
class Mean_Normalisation_Column(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''Perform mean normalization on a specific column'''
        try: 
            column_name = request.args.get('column_name')

            if column_name:
                global DATA_FRAME
                DATA_FRAME = PreProcessing.mean_normalization_column(DATA_FRAME, column_name)
                
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Mean Normalization Column',
                    desc     = 'Performed mean normalization on a specific selected columns',
                    endpoint = '/normalization/mean-normalisation-column',
                    method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column Name to select for mean score normalisation", 
                                    defaultValue= column_name )
                
                api_param_list.append(api_param)
              
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
            
                return APIResponse.success(message = "Performed Mean normalisation on a selected coloumn")
            else:
                return APIResponse.failure(message = "Invalid Argument")
        except ValueError as e:
            app.logger.error("Error: ", e)
            return APIResponse.failure(message=str(e))
        except Exception as e:
            app.loggger.error("Exception : ", e)
            return APIResponse.failure(message=str(e)),500
            

@normalization_ns.route('/min-max-scaling')
@normalization_ns.doc('Perform feature scaling (min-max scaling) on the numeric columns')
class Min_Max_Scaling(Resource):
    def post(self):
        '''
        Perform feature scaling (min-max scaling) on the numeric columns
        '''
        try: 
            global DATA_FRAME
            DATA_FRAME = PreProcessing.min_max_scaling_dataframe(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Min-max scaling',
                    desc     = 'Performed min-max scaling normalisation on a specific selected columns',
                    endpoint = '/normalization/min-max-scaling',
                    method   = 'POST',
                )
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()
        except Exception as e:
            app.logger.error("Exception : ", e)
            return APIResponse.failure(message=str(e)),500
            
@normalization_ns.route('/min-max-scaling-column')
@normalization_ns.doc('Perform feature scaling (min-max scaling) on a specific column of the input DataFrame')
class Min_Max_Scaling_Column(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''
        Perform feature scaling (min-max scaling) on a specific column of the input DataFrame
        '''
        try: 
            column_name = request.args.get('column_name')

            if column_name:
                global DATA_FRAME
                DATA_FRAME = PreProcessing.min_max_scaling_column(DATA_FRAME, column_name)
            
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Min Max Scaling Column',
                    desc     = 'Performed Min max scaling on selected column',
                    endpoint = '/normalization/min-max-scaling-column',
                    method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column Name to select for normalisation", 
                                    defaultValue= column_name )
                
                api_param_list.append(api_param)
              
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()
        
        except ValueError as e:
            app.logger.error("Value Error: ", e)
            return APIResponse.failure(message=str(e)), 400
        except Exception as e:
            app.logger.error("Exception: ", e)
            return APIResponse.failure(message=str(e)),500
            


@normalization_ns.route('/robust-scaling')
@normalization_ns.doc('Perform Robust Scaling on the numeric columns of the input DataFrame.')
class Robust_Scaling(Resource):
    def post(self):
        '''
        Perform Robust Scaling on the numeric columns of the input DataFrame.
        '''
        try: 
            global DATA_FRAME
            DATA_FRAME = PreProcessing.robust_scaling_dataframe(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Robust Scaling',
                    desc     = 'Performed robust scaling on numeric columns',
                    endpoint = '/normalization/robust-scaling',
                    method   = 'POST',
                )
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
                
            return APIResponse.success()
        
        except Exception as e:
            app.logger.error("Exception: ", e)
            return APIResponse.failure(message=str(e)),500
            



@normalization_ns.route('/robust-scaling-column')
@normalization_ns.doc('Perform Robust Scaling on the numeric columns of the input DataFrame.')
class Robust_Scaling_Column(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''
        Performed Robust scaling on the numeric columns of dataframe
        '''
        try: 
            column_name = request.args.get('column_name')

            if column_name:
                global DATA_FRAME
                DATA_FRAME = PreProcessing.robust_scaling_column(DATA_FRAME, column_name)
                
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Robust Scaling Column',
                    desc     = 'Performed robust scaling on selected column',
                    endpoint = '/normalization/robust-scaling-column',
                    method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column Name to select for normalisation", 
                                    defaultValue= column_name )
                
                api_param_list.append(api_param)
              
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
            
                
                return APIResponse.success()
            else:
                return APIResponse.failure(message = "column_name is not valid")
        except ValueError as e:
            app.logger.error("Value Error: ",e)
            return APIResponse.failure(message=str(e))
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message=str(e)),500
            


@normalization_ns.route('/unit-vector-scaling')
@normalization_ns.doc('Perform Unit Vector Scaling (Vector Normalization)')
class Unit_Vector_Scaling(Resource):
    def post(self):
        '''
        Perform Unit Vector Scaling (Vector Normalization)
        '''
        try: 
            global DATA_FRAME
            DATA_FRAME = PreProcessing.unit_vector_scaling_dataframe(DATA_FRAME)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Unit Vector Scaling Column',
                desc     = 'Performed Unit Vector scaling on selected column',
                endpoint = '/normalization/unit-vector-scaling',
                method   = 'POST',
            )
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()
            
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message=str(e)),500
            


@normalization_ns.route('/unit-vector-scaling-column')
@normalization_ns.doc('Perform Unit Vector Scaling (Vector Normalization) on a specific column')
class Unit_Vector_Scaling_Column(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''
        Perform Unit Vector Scaling (Vector Normalization) on a specific column
        '''
        try: 
            column_name = request.args.get('column_name')

            if column_name:
                global DATA_FRAME
                DATA_FRAME = PreProcessing.unit_vector_scaling_column(DATA_FRAME, column_name)
                
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Unit Vector Scaling Column',
                    desc     = 'Performed unit vector scaling on selected column',
                    endpoint = '/normalization/unit-vector-scaling-column',
                    method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column Name to select for normalisation", 
                                    defaultValue= column_name )
                
                api_param_list.append(api_param)
              
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
            
                return APIResponse.success()
            else:
                return APIResponse.failure(message = "Invalid Argument")
        except ValueError as e:
            app.logger.error("Value Error: ",e)
            return APIResponse.failure(message=str(e))
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message=str(e)),500
            

@normalization_ns.route('/label-encode-column')
class LableEncoderColumn(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        """
        Apply LabelEncoder to encode categorical labels in a specific column of a DataFrame
        """
        try:    
            
            column_name = request.args.get('column_name')
            
            global DATA_FRAME
            global LABEL_ENCODER
            DATA_FRAME, LABEL_ENCODER = PreProcessing.label_encode_column(DATA_FRAME, column_name)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Label Encoder',
                desc     = 'Add label encoder to coloumn',
                endpoint = '/normalization/label-encode-column',
                method   = 'POST',
                )
            api_param_list = []
            
            api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='str', 
                                    description="Column Name", 
                                    defaultValue= column_name )
            api_param_list.append(api_param)
            
            api_endpoint.params = api_param_list
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message = "LabelEncoder Applied Successfully"), 200
        except Exception as e:
            app.logger.exception("Exception in Label Encoder: ", str(e)), 500
            return APIResponse.failure(message = str(e)), 500
        
@normalization_ns.route('/label-encode-column/inverse')
class InverseLabelEncoderCol(Resource):
    @normalization_ns.expect(columnName_parser)
    def post(self):
        '''
        Apply inverse transformation to revert label encoded values to original categorical labels in a specific column of a DataFrame
        '''
        try:
            column_name = request.args.get('column_name')
            
            global DATA_FRAME
            DATA_FRAME = PreProcessing.inverse_label_encode_column(df = DATA_FRAME, 
                                                                   column_name = column_name, 
                                                                   label_encoder =LABEL_ENCODER)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Inverse Label Encoder',
                desc     = 'Inverse Label Encoder',
                endpoint = '/normalization/label-encode-column/inverse',
                method   = 'POST',
                )
            api_param_list = []
            
            api_param = APIParam(in_="query", 
                                    name='column_name', 
                                    required=True, 
                                    type_='string', 
                                    description="Column name", 
                                    defaultValue= column_name  
                                 )

            api_param_list.append(api_param)
            api_endpoint.params = api_param_list
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message = "Inverse transformaiton to revert label applied Successfully"), 201
        except Exception as e:
            app.logger.exception("Inverse Label Encoder Column : ",str(e))
            return APIResponse.failure(message = str(e)), 500

@ml_model_ns.route('/split-data')
@ml_model_ns.doc("Splits the dataframe into X (features) and y (target variables)")
class SplitData(Resource):
    @ml_model_ns.expect(split_data_body)
    def post(self):
        '''
        Splits the dataframe into X (features) and y (target variables)
        '''
        body = request.json
        target_columns_list = body.get('target_columns')
        
        if target_columns_list :
            global X, Y 
            X, Y = ML_Models.split_data(DATA_FRAME , target_columns_list)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Split Data',
                desc     = 'Splits dataframe into Features and Target Variables',
                endpoint = '/model/split-data',
                method   = 'POST',
            )
            api_param_list = []
                
            api_param = APIParam(in_="body", 
                                    name='target_columns', 
                                    required=True, 
                                    type_='list', 
                                    description="List of target coloumns name. Ex. ['col1', 'col2']", 
                                    defaultValue= request.json )
                
            api_param_list.append(api_param)
              
            api_endpoint.params = api_param_list
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success()
        else:
            app.logger.error("Request Body: ", body)
            APIResponse.failure(message = 'Invalid Request Body')


@ml_model_ns.route('/dataset/train-test-split-data')
@ml_model_ns.doc("Splits the dataframe into train and test sets")
class SplitDataframeIntoTrainTest(Resource):
    @ml_model_ns.expect(train_test_split_data)
    def post(self):
        try:
            body = request.json
            target_columns_list = body.get('target_columns')
            
            if DATA_FRAME is None:
                return APIResponse.failure(message = "Dataframe not instantiated")
            
            if target_columns_list:
                global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
                X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = ML_Models.train_test_split_data(DATA_FRAME , target_columns_list)
            else:
                app.logger.error("")
                raise CustomExceptions.InvalidRequest(message = "Invalid Request Body")
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(str(e)), 400
        
        api_endpoint = APIEndpoint(
            id       = len(API_HANDLER_LIST)+1,
            name     = 'Dataframe split into test and train',
            desc     = 'Splits the dataframe into train and test sets',
            endpoint = '/model/dataset/train-test-split-data',
            method   = 'POST',
        )
        api_param_list = []
                
        api_param = APIParam(in_="body", 
                                    name='target_columns', 
                                    required=True, 
                                    type_='list', 
                                    description="List of target coloumns name. Ex. ['col1', 'col2']", 
                                    defaultValue= request.json )
                
        api_param_list.append(api_param)
              
        api_endpoint.params = api_param_list
                    
        app.logger.info("Endpoint added to list")
        app.logger.debug("Endpoint added: ", api_endpoint)
        API_HANDLER_LIST.append(api_endpoint.to_dict())
            
        return APIResponse.success(message = "df splitted into X, Y"),200

@ml_model_ns.route('')
@ml_model_ns.doc('Create model')
class Ml_Model(Resource):
    @ml_model_ns.expect(create_model_dto)
    def post(self):
        try:
            body = request.json
            model_type = body.get('model_type')
            if model_type:
                global MODEL
                MODEL = ML_Models.create_model(model_type)
                
                if MODEL is None:
                    return APIResponse.failure(message= 'This model is not suitable for specified')
                
                api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Create Model',
                    desc     = 'Create ML Model',
                    endpoint = '/model',
                    method   = 'POST',
                )
                api_param_list = []
                
                api_param = APIParam(in_="body", 
                                    name='model_type', 
                                    required=True, 
                                    type_='string', 
                                    description="Select model type", 
                                    defaultValue= request.json
                                    )
                
                api_param_list.append(api_param)
              
                api_endpoint.params = api_param_list
                    
                app.logger.info("Endpoint added to list")
                app.logger.debug("Endpoint added: ", api_endpoint)
                API_HANDLER_LIST.append(api_endpoint.to_dict())
            
                return APIResponse.success(message = "Model Created successfully"),201
            
            else:
                raise CustomExceptions.InvalidRequest(message="model_type is not provided in request body")
        except CustomExceptions.InvalidRequest as e:
            app.logger.error("Invalid Request: ",e)
            return APIResponse.failure(message = str(e))
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message = str(e))
        
@ml_model_ns.route('/train/fit-model')
class Fit_Model(Resource):
    
    def post(self):
        '''
        Fits a machine learning model to the provided training data
        '''
        global MODEL
        if MODEL is None:
            return APIResponse.failure(message = "model is not created"), 400
        if X_TRAIN is None :
            return APIResponse.failure(message = "X_Train is not initialised"), 400
        if Y_TRAIN is None:
            return APIResponse.failure(message = "Y_Train is not initialised"), 400
        
        try:
            MODEL = ML_Models.fit_model(MODEL, X_TRAIN, Y_TRAIN)
            app.logger.info("Model after fit: ", str(MODEL))
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Fit Model',
                desc     = 'Fits a machine learning model to the provided training data',
                endpoint = '/model/train/fit-model',
                method   = 'POST',
            )
            
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
                        
            return APIResponse.success()
            
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message = str(e))
            
@ml_model_ns.route('/train/predict-with-model')
class Predict_Model(Resource):
    
    def post(self):
        '''
        Predicts target variables using a pre-trained model
        '''
        if MODEL is None :
            app.logger.debug("MODEL: ", MODEL)
            return APIResponse.failure(message="Model is not created"), 400
        
        try:
            global PREDICTIONS
            PREDICTIONS = ML_Models.predict_with_model(MODEL, X_TEST)
            
            
            api_endpoint = APIEndpoint(
                    id       = len(API_HANDLER_LIST)+1,
                    name     = 'Predict Model',
                    desc     = 'Predicts target variables using a pre-trained model',
                    endpoint = '/model/train/predict-with-model',
                    method   = 'POST',
            )
           
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            app.logger.info('Predictions : ', PREDICTIONS)
            return APIResponse.success(data = {"predictions": str(PREDICTIONS)}) 
    
        except Exception as e:
            app.logger.error("Exception: ",str(e))
            return APIResponse.failure(message = str(e))
             
@ml_model_ns.route("/train/evaluate-predictions")
class EvaluatePredictions(Resource):
    
    def post(self):
        if MODEL is None:
            app.logger.debug("MODEL: ", str(MODEL))
            return APIResponse.failure(message = "ML-MODEL not created"),400
        
        try: 
            metrics = ML_Models.evaluate_predictions(y_true=Y_TEST, y_pred = PREDICTIONS)
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Evaluate Predictions',
                desc     = 'Evaluated Predictions',
                endpoint = '/model/train/evaluate-predictions',
                method   = 'POST',
            )
           
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(data = {"metrics": metrics}, message= "Predictions are evaluated successfully"), 200
        except Exception as e:
            app.logger.error("Exception: ",e)
            return APIResponse.failure(message = str(e)), 500
        
@ml_model_ns.route("/confusion-matrix")
class ConfusionMatrix(Resource):
    @ml_model_ns.expect(columnName_parser)
    def post(self):
        '''
        Plots Confusion Matrix and save as .png
        '''
        try:  
            target_column = request.args.get('column_name')
            list_unique_ele = PreProcessing.get_unique_elements(DATA_FRAME, target_column)
            
            ML_Models.plot_confusion_matrix(Y_TEST, PREDICTIONS, classes=list_unique_ele )
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Generate Confusion Matrix',
                desc     = 'Generate Confusion Matrix',
                endpoint = '/model/plot/confusion-matrix',
                method   = 'POST',
                )
            api_param_list = []
    
            api_endpoint.params = api_param_list
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(message = "Confusion Matrix image is saved as confustion_matrix.png")
        except Exception as e:
            app.logger.exception("Exception in plotting confusion matrix: ", e)
            return APIResponse.failure(message=str(e))

    def get(self):
        '''
        Returns Confusion Matix Image
        '''
        try:
            # confusion_matrix_img_path = 'D:\\NCP(No Code Platform)\\ncp-backend(python)\\confusion_matrix.png'
            filename = "confusion_matrix.png"
            img_path = os.path.join(os.getcwd(), filename)
            
            return APIResponse.success(data = convertToBase64(img_path) )
        except Exception as e:
            app.logger.warning("Exception in Confusion Matrix Image: ", str(e))
            return APIResponse.failure(message = "File Not Found"), 404
@ml_model_ns.route("/save")
class SaveModel(Resource):
    @ml_model_ns.expect(userId_filename_dto)
    def post(self):
        try:
            body = request.json
            
            userId = body.get('userId')
            filename = body.get('filename')
            
            user = Users.query.filter_by(id = userId).first()
            
            file_path = f'{UPLOAD_FOLDER}/{get_folder_name(user)}/ml-models'
            
            # if models directory not present then create the directory
            if not os.path.exists(file_path):
                os.mkdir(os.path.join(file_path))
            
            saved_file_path = ML_Models.save_model(MODEL, file_path, filename)
            
            app.logger.info("Model Saved successfully")
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Save Model',
                desc     = f'Model saved at {saved_file_path}',
                endpoint = '/model/save',
                method   = 'POST',
            )
            api_param_list = []
    
            api_endpoint.params = api_param_list
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(data = {"filepath" : saved_file_path}, message = "Model saved successfully"), 201
        except Exception as e: 
            app.logger.exception("Exception while saving the model : ", str(e))
            return APIResponse.failure(message = str(e)), 500

@ml_model_ns.route("/save/pickle")
class SavePickle(Resource):
    @ml_model_ns.expect(userId_filename_dto)
    def post(self):
        try:
            body = request.json
            
            userId = body.get('userId')
            filename = body.get('filename')
            
            user = Users.query.filter_by(id = userId).first()
            
            file_path = f'{UPLOAD_FOLDER}/{get_folder_name(user)}/ml-models-pickle'
            
            # if models-pickle directory not present then create the directory
            if not os.path.exists(file_path):
                os.mkdir(os.path.join(file_path))

            saved_file_path = ML_Models.save_model_as_pickle(MODEL, file_path, filename)
            
            
            api_endpoint = APIEndpoint(
                id       = len(API_HANDLER_LIST)+1,
                name     = 'Save Pickel',
                desc     = 'Pickel File is generated',
                endpoint = '/model/save/pickle',
                method   = 'POST',
                )
                    
            app.logger.info("Endpoint added to list")
            app.logger.debug("Endpoint added: ", api_endpoint)
            API_HANDLER_LIST.append(api_endpoint.to_dict())
            
            return APIResponse.success(data = {"filepath": saved_file_path}, message = "Pickle file saved successfully"), 201
        except Exception as e: 
            app.logger.exception("Error in saving pickle file : ", str(e))
            return APIResponse.failure(data = str(e)), 500
@supervised_learning_types_ns.route('')
#/supervised-learning
class SupervisedLearningTypes(Resource):
    @supervised_learning_types_ns.expect(supervised_learning_algo_dto)
    def post(self):
        '''Add Supervised Learning Types'''
        data = request.json
        new_algorithm = SupervisedLearningAlgorithms(
            name=data['name'],
            value=data['value'],
            category=data['category']
        )
        try:
            db.session.add(new_algorithm)
            db.session.commit()
            return APIResponse.success(data = new_algorithm.to_dict(),
                                       message = "New Algorithm Added Successfully"), 201
        except Exception as e:
            db.session.rollback()
            return APIResponse.failure(message = str(e)), 500
        
    def get(self):
        '''Get Supervised Learning Types'''
        algorithms = SupervisedLearningAlgorithms.query.all()
        return APIResponse.success(data = algorithms)
        # return jsonify([algorithm.to_dict() for algorithm in algorithms])
        
        
    def put(self):
        '''Edit Supervised learning types'''
    
    def delete(self):
        '''Delete Supervised Learning types'''
        try:
            algorithm = SupervisedLearningAlgorithms.query.filter_by(id=1).one()
            
            db.session.delete(algorithm)
            db.session.commit()
            
            return APIResponse.success(message = 'Algorithm deleted'),201
        except Exception as e:
            return APIResponse.failure(message = "Not found in db"), 404    

@history_tracker_ns.route('/operations-performed')
class GetFileLogsHistory(Resource):
    def get(self):
        return APIResponse.success(data = API_HANDLER_LIST, 
                                   message = "History of given file "), 200

@visualisation_ns.route('/boxplot')
class BoxPlot(Resource):
    @visualisation_ns.expect(boxplot_body_dto)
    def post(self):
        try:
            body = request.json
            figsize = body.get("figsize")
            
            
            filename = Visualization.generate_boxplot(data= DATA_FRAME, 
                                           x_label = body.get('x_label') , 
                                           y_label = body.get('y_label') , 
                                           title = body.get('title') ,
                                           xticks_rotation= body.get('xticks_rotation'), 
                                           figsize = (figsize.get('width'), figsize.get('height')))
            
            # boxplot_img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            
            img_path = os.path.join(os.getcwd(), filename)
            
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
                base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
            # Additional parameters
            additional_params = {
                "image": base64_img
            }
        
            return APIResponse.success(data = additional_params )
            
            
            # return send_file(path_or_file= boxplot_img_path, mimetype= 'image/png')
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500


@visualisation_ns.route('/plot-correlation-matrix')
class PlotCorrelationMatrix(Resource):
    @visualisation_ns.expect(plot_correlation_matrix_dto)
    def post(self):
        try:
            body = request.json
            figsize = body.get("figsize")
            
            
            filename = Visualization.plot_correlation_matrix(df = DATA_FRAME, 
                                                             method = body.get('method'),
                                                             annot = body.get('annot'), 
                                                             cmap = body.get('coolwarm'), 
                                                             figsize = (figsize.get('width'), figsize.get('height'))
                                                             ) 
                                                             
            img_path = os.path.join(os.getcwd(), filename)
            
            return APIResponse.success(data = convertToBase64(img_path) )
            
            
            # return send_file(path_or_file= img_path, mimetype= 'image/png')
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500

@visualisation_ns.route('/barchart')
class BarChart(Resource):
    @visualisation_ns.expect(barchart_dto)
    def post(self):
        try:
            body = request.json
            
            filename = Visualization.create_bar_chart(categories=body.get('categories'),
                                                      values = body.get('values'),
                                                      title = body.get('title'),
                                                      xlabel = body.get('xlabel'),
                                                      ylabel = body.get('ylabel'),
                                                      filename = None)
            
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path))
            
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500

@visualisation_ns.route('/bar-chart-from-dataframe')
class BarChartFromDataframe(Resource):
    @visualisation_ns.expect(barchart_df_dto)
    def post(self):
        try:
            body = request.json
            
            filename = Visualization.create_bar_chart_from_dataframe( dataframe = DATA_FRAME,
                                                                     x_columns = body.get('x_columns'),
                                                                     y_columns = body.get('y_columns'),
                                                                     title = body.get('title'),
                                                                     xlabel = body.get('xlabel'),
                                                                     ylabel = body.get('ylabel'),
                                                                     filename = None)
            
            
            # barchart_img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path) )
            
            # return send_file(path_or_file= boxplot_img_path, mimetype= 'image/png')
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.warning('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500


@visualisation_ns.route('/scatterplot/custom-scatter-plot')
class CustomScatterPlot(Resource):
    @visualisation_ns.expect(cus_scatter_plt_dto)
    def post(self):
        try:
            body = request.json
            
            filename = Visualization.custom_scatterplot( df = DATA_FRAME,
                                                        x_col = body.get('x_col'),
                                                        y_col = body.get('y_col'),
                                                        hue_col = body.get('hue_col'),
                                                        x_label = body.get('x_label'),
                                                        y_label = body.get('y_label'),
                                                        title = body.get('title')
                                                        )
            
            
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path) )
                    
            # return send_file(path_or_file= img_path, mimetype= 'image/png')
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('custom_scatterplot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500
 

@visualisation_ns.route('/scatterplot/scatterplot-each-column')
class ScatterPlotEachCol(Resource):
    @visualisation_ns.expect(scatterplot_col_dto)
    def post(self):
        try:
            body = request.json
            
            filepath_list = Visualization.scatterplot_each_column( df = DATA_FRAME,
                                                             target_column= body.get('target_column'),
                                                             hue_column = body.get('hue_column'),
                                                             hue_order = body.get('hue_order'),
                                                            )
            
            return APIResponse.success(data = filepath_list), 201
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500
    
    @visualisation_ns.expect(filename_parser)
    def get(self):
        '''Get Scatter Plot Image as per filename'''
        try:
            filename = request.args.get('filename')
            
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path) )
        
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as e:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500
        


@visualisation_ns.route('/plot-dataframe-desc')
class PlotDataframeDesc(Resource):
    @visualisation_ns.expect(plot_df_dto)
    def post(self):
        try:
            body = request.json
            figsize = body.get('figsize')
            
            filename = Visualization.plot_dataframe_description(df = DATA_FRAME, 
                                                                figsize = (figsize.get('width'), figsize.get('height')),
                                                                bar_width = body.get('bar_width'),
                                                                error_bars = body.get('error_bars'),
                                                                )
            
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            
            return APIResponse.success(data = convertToBase64(img_path) )
    
            # return send_file(path_or_file= img_path, mimetype= 'image/png')
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            app.logger.warning('Value Error: ', ve)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500


@visualisation_ns.route('/plot-stats-summary')
class PlotStatsSummary(Resource):
    # @visualisation_ns.expect(plot_stats_dto)
    def post(self):
        '''Plot the summary statistics of a pandas DataFrame'''
        try:
            filename = Visualization.plot_statistics_summary( df = DATA_FRAME)
            
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path) )
            
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except ValueError as ve:
            app.logger.info('Request Body: ', request.json)
            return APIResponse.failure(message = 'Invalid Request Body'), 400
        except Exception as e: 
            app.logger.exception('BoxPlot Exceptions: ', str(e))
            return APIResponse.failure(message = 'Server Issue'), 500

global TRANSFORMED_DATA, PCA_MODEL
@pca_ns.route('/apply-pca/<int:num_components>')
class PCA(Resource):
    def post(self, num_components):
        '''Apply PCA (Principal Component Analysis) to a dataset(2D data)'''
        try:
            global TRANSFORMED_DATA, PCA_MODEL
            TRANSFORMED_DATA, PCA_MODEL = Pca.apply_pca(dataset = DATA_FRAME, n_components=num_components)
            
            return APIResponse.success(message = "PCA Applied successfully"), 201
        except Exception as e:
            app.logger.warning('Apply PCA exception: ', e)
            return APIResponse.failure(message = 'Operation failed'), 500
    

@pca_ns.route('/apply-pca-dataframe/<int:num_components>')
class PCA_Df(Resource):
    def post(self, num_components):
        '''Apply PCA (Principal Component Analysis) to a dataset where input dataset is DATAFRAME'''
        try:
            global TRANSFORMED_DATA, PCA_MODEL
            TRANSFORMED_DATA, PCA_MODEL = Pca.apply_pca_Dataframe(dataset = DATA_FRAME, n_components=float(num_components))
            
            return APIResponse.success(message = "PCA Applied successfully"), 201
        except Exception as e:
            app.logger.warning('Apply PCA exception: ', e)
            return APIResponse.failure(message = 'Operation failed'), 500

global RECONSTRUCT_DATA, RECONSTRUCT_DF
@pca_ns.route('/reconstruct-data')
class ReconstructData(Resource):
    def post(self):
        '''Reconstruct the original data from compressed data using PCA'''
        try:
            global RECONSTRUCT_DATA
            RECONSTRUCT_DATA = Pca.reconstruct_data(compressed_data = TRANSFORMED_DATA,
                                                        pca_model   = PCA_MODEL)
            
            return APIResponse.success(message = "Reconstructed Original Data"), 201
        except Exception as e:
            app.logger.warning('Reconstructed Data: ', e)
            return APIResponse.failure(message = 'Operation failed'), 500

@pca_ns.route('/reconstruct-data-df')
class ReconsDataDf(Resource):
    def post(self):
        '''
            Reconstruct the original data from compressed data using PCA.
        '''
        try:
            global RECONSTRUCT_DF
            RECONSTRUCT_DF = Pca.reconstruct_data_dataframe(compressed_data= TRANSFORMED_DATA, 
                                                                         pca_model= PCA_MODEL)
            
            return APIResponse.success(message = "Reconstructed original data (DataFrame)"), 201
        except Exception as e:
            app.logger.warning('Reconstruct Original Data: ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/compress-reconstruct/<int:num_components>')
class CompressReconstruct(Resource):
    def post(self , num_components):
        '''
            Compress the dataset using PCA and then reconstruct it back to the original form.
        '''
        try:
            global TRANSFORMED_DATA, RECONSTRUCT_DATA
            TRANSFORMED_DATA, RECONSTRUCT_DATA = Pca.compress_and_reconstruct(dataset = DATA_FRAME, n_components= num_components)
            
            return APIResponse.success(message = "Compress and Reconstructed Data is generated successfully"), 201
        except Exception as e:
            app.logger.warning('Compress and Reconstructed Data: ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/compress-reconstruct-df/<int:num_components>')
class CompressReconstructDf(Resource):
    def post(self, num_components):
        '''
            Compress the dataset using PCA and then reconstruct it back to the original form.
        '''
        try:
            global TRANSFORMED_DATA,RECONSTRUCT_DATA
            TRANSFORMED_DATA, RECONSTRUCT_DATA = Pca.compress_and_reconstruct_dataframe(dataset = DATA_FRAME, n_components= num_components)
            
            return APIResponse.success(message = "Reconstructed original data (DataFrame)"), 201
        except Exception as e:
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500

@pca_ns.route('/variance-ratio')
class VarianceRatio(Resource):
    def get(self):
        '''
         Calculate explained variance ratio for each principal component.
        '''
        try:
            explained_variance_ratio = Pca.explained_variance_ratio(pca_model= PCA_MODEL)
            # print("VARIANCE RATIO: ",type(explained_variance_ratio))
            # print(str(explained_variance_ratio))
            # # list issue with return type with toList()
            return APIResponse.success(data = {'variance_ratio': "explained_variance_ratio"},
                                       ), 200
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500

@pca_ns.route('/optimal-num-components')
class ChooseNComponents(Resource):
    @pca_ns.expect(threshold_parser)
    def get(self):
        '''
        Get the number of components based on a threshold of explained variance.
        '''
        try:
            threshold = request.args.get('threshold')
            n_components = Pca.choose_n_components(dataset = DATA_FRAME, 
                                                   threshold = threshold)
            
            return APIResponse.success(data = {'n_components': n_components}), 200
        
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/optimal-num-components-df')
class ChooseNComponentsDf(Resource):
    @pca_ns.expect(threshold_parser)
    def get(self):    
        try:
            threshold = request.args.get('threshold')
            n_components = Pca.choose_n_components_dataframe(dataset = DATA_FRAME, 
                                                            threshold = threshold)
            
            return APIResponse.success(data = {'n_components': n_components}), 200
        
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500
    

@pca_ns.route('/plot-scree')
class PlotScree(Resource):
    def post(self):
        ''' Plot scree plot showing the explained variance for each principal component'''
       
        try:
            filename = Pca.plot_scree(pca_model= PCA_MODEL)
        
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path) )
            
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/plot-cumulative-variance')
class PlotCumulativeVariance(Resource):
    def post(self):
        '''  Plot cumulative explained variance'''
       
        try:
            filename = Pca.plot_cumulative_variance(pca_model= PCA_MODEL)
        
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path))
               
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500

@pca_ns.route('/biplot')
class Biplot(Resource):
    @pca_ns.expect(biplot_dto)
    def post(self):
        ''' Plot scree plot showing the explained variance for each principal component'''

        try:
            body = request.json
            if 'labels' in body:
                labels = body.get('labels')
            else:
                labels = None
                
            filename = Pca.biplot(pca_model= PCA_MODEL,
                                #   Which data should be passed confirm it 
                                  data = TRANSFORMED_DATA,
                                  labels = labels 
                                )
        
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path))
            # return send_file(path_or_file= img_path, mimetype= 'image/png')
        
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/biplot-dataframe')
class BiplotDataframe(Resource):
    @pca_ns.expect(biplot_dto)
    def post(self):
        ''' Plot scree plot showing the explained variance for each principal component'''

        try:
            body = request.json
            if 'labels' in body:
                labels = body.get('labels')
            else:
                labels = None
                
            filename = Pca.biplot_dataframe(pca_model= PCA_MODEL,
                                #   Which data should be passed confirm it 
                                  data = TRANSFORMED_DATA,
                                  labels = labels 
                                )
        
            # img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\{filename}'
            img_path = os.path.join(os.getcwd(), filename)
            return APIResponse.success(data = convertToBase64(img_path))
            # return send_file(path_or_file= img_path, mimetype= 'image/png')
        
        except FileNotFoundError as e:
            return APIResponse.failure(message = 'File Not Found'), 404
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e: 
            app.logger.warning('Exception ', e)
            return APIResponse.failure(message = 'Operation failed'), 500


@pca_ns.route('/remove-outliers-pca')
class RemoveOutliersPCA(Resource):
    @pca_ns.expect(threshold_parser)
    def post(self):
        '''
        Remove outliers from the dataset based on the distance from the centroid in PCA space.        
        '''
        try:
            threshold = request.args.get('threshold')
            
            dataset = Pca.remove_outliers_pca(dataset= DATA_FRAME, 
                                                    pca_model = PCA_MODEL, 
                                                    threshold = threshold)
            
            return APIResponse.success(message = 'Outliers removed from df')
        except Exception as e :
            app.logger.warning('Remove Outliers PCA: ', str(e))
            return APIResponse.failure(message = 'Operation failed'), 500   



@pca_ns.route('/remove-outliers-pca-df')
class RemoveOutliersPcaDf(Resource):
    @pca_ns.expect(threshold_parser)
    def post(self):
        '''
        Remove outliers from the dataset based on the distance from the centroid in PCA space.        
        '''
        try:
            threshold = request.args.get('threshold')
            
            dataset, mask = Pca.remove_outliers_pca_dataframe(dataset = DATA_FRAME, 
                                                                pca_model = PCA_MODEL, 
                                                                threshold = threshold)
            
            return APIResponse.success(message = 'Outliers removed from df')
        except Exception as e :
            return APIResponse.failure(message = 'Operation failed'), 500   


global PRINCIPAL_COMPONENT_DF
@pca_ns.route('/principal-components')
class PrincipalComponents(Resource):
    def get(self):
        '''
        Retrieve individual principal components from a PCA model.
        '''
        try: 
            global PRINCIPAL_COMPONENT_DF
            PRINCIPAL_COMPONENT_DF = Pca.get_principal_components(pca_model= PCA_MODEL)
            
            return APIResponse.success(message = 'Principal Component Df generated successfully')
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e :
            return APIResponse.failure(message = 'Operation failed'), 500   


@pca_ns.route('/inverse-transform-component')
class InverseTransformComponent(Resource):
    def post(self):
        '''
        Perform inverse transformation for an individual principal component.
        '''
        try:
            ndarry = Pca.inverse_transform_component(component = PRINCIPAL_COMPONENT_DF, 
                                                    pca_model = PCA_MODEL)
            return APIResponse.success(message = 'Reconstructed feature vector generated successfully')
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e :
            app.logger.warning('Inverse Transform Component Exception: ', str(e))
            return APIResponse.failure(message = 'Operation failed'), 500   


@pca_ns.route('/inverse-transform-component-df')
class InverseTransformComponentDf(Resource):
    def post(self):
        '''
        Perform inverse transformation for an individual principal component.
        '''
        try:
            ndarry_df = Pca.inverse_transform_component_dataframe(component = PRINCIPAL_COMPONENT_DF, 
                                                                  pca_model = PCA_MODEL)
            return APIResponse.success(message = 'Reconstructed feature vector df generated successfully')
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e :
            app.logger.warning('Inverse Transform Component Exception: ', str(e))
            return APIResponse.failure(message = 'Operation failed'), 500   


@pca_ns.route('/cross-validation')
class CrossValidation(Resource):
    @pca_ns.expect(cross_validation_dto)
    def post(self):
        ''' 
        Perform cross-validation for PCA.
        '''
        try:
            body = request.json
            
            cross_validation_scores_arr = Pca.cross_val_pca(dataset=DATA_FRAME, 
                              estimator = MODEL, 
                              cv = body.get('cross_validation'))
            return APIResponse.success(data = cross_validation_scores_arr,
                                       message = 'Cross validation scores generated successfully')
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Cross Validation Exception ', str(e))
            return APIResponse.failure(message = str(e)), 500

@pca_ns.route('/cross-validation-df')
class CrossValidationDf(Resource):
    @pca_ns.expect(cross_validation_dto)
    def post(self):
        ''' 
        Perform cross-validation for PCA.
        '''
        try:
            body = request.json
            
            cross_validation_scores_arr = Pca.cross_val_pca_dataframe(
                            dataset=DATA_FRAME, 
                            estimator = MODEL, 
                            cv = body.get('cross_validation'))
            return APIResponse.success(data = cross_validation_scores_arr,
                                       message = 'Cross validation scores df generated successfully')
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Cross Validation Exception ', str(e))
            return APIResponse.failure(message = str(e)), 500

            
@pca_ns.route('/feature-selection')
class FeatureSelection(Resource):
    
    @pca_ns.expect(num_components_parser)
    def post(self):
        '''
        Perform PCA for feature selection.
        '''
        try:
            num_components = request.args.get('num_components')
            x_pca , selected_features = Pca.feature_selection_pca(
                dataset = DATA_FRAME, 
                n_components= num_components,
            )
            return APIResponse.success(message = 'Transformed dataset and selected features generated successfully'), 201
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Feature Selection Exception : ', str(e))
            return APIResponse.failure(message = str(e)), 500
        

@pca_ns.route('/feature-selection-df')
class FeatureSelectionDf(Resource):
    
    @pca_ns.expect(num_components_parser)
    def post(self):
        '''
        Perform PCA for feature selection.
        '''
        try:
            num_components = request.args.get('num_components')
            x_pca , selected_features = Pca.feature_selection_pca_dataframe(
                dataset = DATA_FRAME, 
                n_components= num_components,
            )
            return APIResponse.success(message = 'Transformed dataset and selected features generated successfully'), 201
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Feature Selection Df Exception : ', str(e))
            return APIResponse.failure(message = str(e)), 500


@pca_ns.route('/cluster-visualization')
class ClusterVisualization(Resource):
    
    def post(self):
        '''
        Visualize clusters in reduced PCA space and save the plot in the current working directory.
        '''
        try:
            # Get target column from earlier api
            # find unique values from target columns and assign to labels
            # else default value from target cols unique values.
            
            # Pca.cluster_visualization_pca(dataset = DATA_FRAME, 
                                        #   labels =  )
            
            return APIResponse.success(message = 'Transformed dataset and selected features generated successfully'), 201
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Feature Selection Exception : ', str(e))
            return APIResponse.failure(message = str(e)), 500

@pca_ns.route('/cluster-visualization-df')
class ClusterVisualizationDf(Resource):
    def post(self):
        '''
        Visualize clusters in reduced PCA space and save the plot in the current working directory.
        '''
        try:
            # Get target column from earlier api
            # find unique values from target columns and assign to labels
            # else default value from target cols unique values.
            
            # Pca.cluster_visualization_pca_dataframe(dataset = DATA_FRAME,
            #                                         labels = "",
            #                                         pca_model = PCA_MODEL,   
            #                                         )
            
            return APIResponse.success(message = 'Transformed dataset and selected features generated successfully'), 201
        except NameError as e:
            return APIResponse.failure(message = str(e)), 500
        except Exception as e:
            app.logger.warning('Feature Selection Exception : ', str(e))
            return APIResponse.failure(message = str(e)), 500

@kmeans_ns.route('/clustering')
class KMeansClustering(Resource):
    @kmeans_ns.expect(kmeans_clustering_parser)
    def post(self):
        '''
            Perform k-means clustering on the given dataframe.
        '''
        try:
            nos_of_cluster = request.args.get("nos_of_cluster")
            KMEAN_LABELS, KMEAN_CENTROIDS = Kmeans.kmeans_clustering(dataframe= DATA_FRAME, 
                                                                     k = nos_of_cluster)
            return APIResponse.success(data = [KMEAN_CENTROIDS, KMEAN_LABELS]), 201
        except Exception as e:
            return APIResponse.failure(message = str(e)), 400
    
    


@app.get("/testing/hello")
def testing():
        # boxplot_img_path = f'D:\\NCP(No Code Platform)\\ncp-backend(python)\\boxplot.png'
        filename = 'boxplot.png'
        img_path = os.path.join(os.getcwd(), filename)
        
        # Read image file
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
            base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Additional parameters
        additional_params = {
            "image": base64_img
        }
        
        return APIResponse.success(data = additional_params )     
if __name__ == '__main__':
    app.run(host=os.getenv('HOST'),port=os.getenv('PORT'), debug=True)
    
