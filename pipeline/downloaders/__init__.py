#!/usr/bin/env python3
"""
Data Downloaders Module

Handles all data acquisition including:
- Web scraping (Spanish news, academic papers)
- API downloads (BOE, HuggingFace, etc.)
- Direct downloads (Wikipedia dumps, etc.)
"""

from .web_scraper import WebScrapingDownloader
from .api_downloader import APIDownloader
from .direct_downloader import DirectDownloader
from .download_orchestrator import DownloadOrchestrator

__all__ = [
    "WebScrapingDownloader",
    "APIDownloader",
    "DirectDownloader",
    "DownloadOrchestrator"
]
