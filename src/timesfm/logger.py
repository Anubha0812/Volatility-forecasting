import logging
# import sys

def get_logger(log_name: str, log_file: str):
  logger = logging.getLogger(log_name)
  logger.setLevel(logging.INFO)
  # Stop propagation so root handlers (console) don't also print messages.
  logger.propagate = False

  # Remove any StreamHandlers attached to this logger (avoid console output).
  for h in list(logger.handlers):
    if isinstance(h, logging.StreamHandler):
      logger.removeHandler(h)

  # Add FileHandler if not already present for this file path.
  file_handler_exists = any(
      isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file
      for h in logger.handlers
  )
  if not file_handler_exists:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
  
  return logger