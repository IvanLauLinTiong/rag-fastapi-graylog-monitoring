import logging
import graypy

logger = logging.getLogger("rag-logger")

# Graylog server configuration
handler = graypy.GELFUDPHandler('localhost', 12201)

logger.setLevel(logging.INFO)
logger.addHandler(handler)
