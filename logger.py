import logging
import graypy

logger = logging.getLogger(__name__)

# Graylog server configuration
handler = graypy.GELFUDPHandler('localhost', 12201)

logger.addHandler(handler)
