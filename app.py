import logging

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Use: streamlit run app/streamlit_app.py")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()

