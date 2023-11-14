

from apiflask import HTTPTokenAuth

auth = HTTPTokenAuth(scheme='ApiKeyAuth')


tokens = {
    "secret-token-1": "john",
    "secret-token-2": "susan"
}


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]
