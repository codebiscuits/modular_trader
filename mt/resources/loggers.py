import logging
from datetime import datetime, timezone
import sys
from pathlib import Path
from logging import handlers
from mt.resources import keys

def create_logger(source, other_path=None):
    day = datetime.now(timezone.utc).strftime('%-d')
    month = datetime.now(timezone.utc).strftime('%m')
    year = datetime.now(timezone.utc).strftime('%y')
    hour = datetime.now(timezone.utc).hour
    folder = Path(f"/home/ross/coding/modular_trader/logs/{year}/{month}/{day}")
    folder.mkdir(parents=True, exist_ok=True)

    def filter_maker(levels):
        levels = [getattr(logging, level) for level in levels]

        def filter(record):
            return record.levelno in levels

        return filter

    logger = logging.getLogger(source)
    logger.setLevel(logging.DEBUG)

    debug_formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(message)s', datefmt='%H:%M %d-%m-%Y')
    debug_handler = logging.StreamHandler(sys.stdout)
    debug_handler.setFormatter(debug_formatter)
    debug_handler.setLevel(logging.DEBUG)
    debug_filter = filter_maker(['DEBUG', 'WARNING', 'ERROR', 'CRITICAL'])
    debug_handler.addFilter(debug_filter)
    logger.addHandler(debug_handler)

    info_formatter = logging.Formatter(fmt='%(message)s')
    info_path = f'{hour}_{other_path}_info.log' if other_path else f'{hour}_info.log'
    info_handler = logging.FileHandler(folder / info_path)
    info_handler.setFormatter(info_formatter)
    info_handler.setLevel(logging.INFO)
    info_filter = filter_maker(['INFO'])
    info_handler.addFilter(info_filter)
    logger.addHandler(info_handler)

    error_formatter = logging.Formatter(fmt='%(asctime)s | %(name)s line %(lineno)d | %(message)s', datefmt='%H:%M %d-%m-%Y')
    error_path = f'{hour}_{other_path}_error.log' if other_path else f'{hour}_error.log'
    error_handler = logging.FileHandler(folder / error_path)
    error_handler.setFormatter(error_formatter)
    error_handler.setLevel(logging.ERROR) # no filter so this means ERROR and above
    logger.addHandler(error_handler)

    # email_handler = handlers.SMTPHandler(
    #     mailhost=('smtp.gmail.com', 465),
    #     fromaddr=keys.gm_adr,
    #     toaddrs=[keys.gm_adr],
    #     subject='Error/Exception from Your Application',
    #     credentials=(keys.gm_adr, keys.gm_pss),
    # )
    # email_handler.setLevel(logging.ERROR)
    # logger.addHandler(email_handler)

    return logger

