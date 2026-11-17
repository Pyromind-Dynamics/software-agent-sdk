from pyromind_agent_server.bootstrap import install_product_api
from pyromind_agent_server.config import Config
from pyromind_agent_server.http_app import create_app as create_http_app


def create_app(config: Config | None = None):
    return install_product_api(create_http_app(config))


api = create_app()
