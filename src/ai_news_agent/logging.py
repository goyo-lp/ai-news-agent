"""Structured logging configuration."""

import logging
import sys

import structlog


def configure_logging(*, level: str = "INFO", output_format: str = "json") -> None:
    """Configure standard-library and structlog output once per process."""

    renderer = (
        structlog.processors.JSONRenderer()
        if output_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level),
        stream=sys.stderr,
        force=True,
    )
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
