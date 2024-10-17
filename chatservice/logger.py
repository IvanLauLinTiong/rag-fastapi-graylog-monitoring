import logging
import graypy

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("rag-logger")

# Graylog server configuration
handler = graypy.GELFUDPHandler('graylog', 12201)

logger.setLevel(logging.INFO)
logger.addHandler(handler)
