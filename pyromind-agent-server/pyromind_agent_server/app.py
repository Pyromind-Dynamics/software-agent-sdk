from openhands.agent_server.api import create_app as create_openhands_app
from openhands.agent_server.config import Config
from pyromind_agent_server.bootstrap import install_product_api


def create_app(config: Config | None = None):
    return install_product_api(create_openhands_app(config))


api = create_app()
