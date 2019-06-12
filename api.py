from flask import Flask
from flask_restful import Api, Resource, reqparse
import json

from cleaner import convert
from extracter import extract
app = Flask(__name__)
api = Api(app)

class SigParser(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("content")
        args = parser.parse_args()

        cleaned_signature = convert(args["content"])
        parsed_signature = extract(cleaned_signature)
        print("parsed_signature", parsed_signature)
        return parsed_signature, 201

api.add_resource(SigParser, "/signature/parse")
app.run(debug=True)


