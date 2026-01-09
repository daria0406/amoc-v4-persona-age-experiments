import logging


def setup_logging(debug: bool):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
