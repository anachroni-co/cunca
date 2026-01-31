from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime
from .version_manager import VersionManager
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)

@dataclass
class ModelRequest:
    model_id: str
    version: Optional[str]
    input_data: Dict[str, Any]
    priority: int = 1
    timeout: int = 30
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ModelRouter:
    def __init__(self, version_manager: VersionManager, resource_manager: ResourceManager):
        self.version_manager = version_manager
        self.resource_manager = resource_manager
        self.request_queue: asyncio.Queue[ModelRequest] = asyncio.Queue()
        self.active_requests: Dict[str, ModelRequest] = {}
        self.request_history: List[Dict[str, Any]] = []

    async def route_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Route a request to a specific model."""
        try:
            # Verify version
            if request.version is None:
                request.version = await self.version_manager.get_active_version(request.model_id)
                if request.version is None:
                    raise ValueError(f"No active version for model {request.model_id}")

            # Verify resources
            await self.resource_manager.update_usage(request.model_id)

            # Process request
            result = await self._process_request(request)

            # Record in history
            self.request_history.append({
                "model_id": request.model_id,
                "version": request.version,
                "timestamp": datetime.now(),
                "status": "success"
            })

            return result

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            self.request_history.append({
                "model_id": request.model_id,
                "version": request.version,
                "timestamp": datetime.now(),
                "status": "error",
                "error": str(e)
            })
            raise

    async def _process_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Process a model request."""
        # Here the actual processing logic would be implemented
        # For now, we simulate a response
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "status": "success",
            "model_id": request.model_id,
            "version": request.version,
            "result": "Simulated response"
        }

    async def get_request_history(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get request history."""
        if model_id:
            return [req for req in self.request_history if req["model_id"] == model_id]
        return self.request_history

    async def get_active_requests(self) -> Dict[str, ModelRequest]:
        """Get active requests."""
        return self.active_requests

    async def cancel_request(self, request_id: str) -> None:
        """Cancel an active request."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            logger.info(f"Request {request_id} cancelled")

    async def monitor_requests(self, interval: int = 60) -> None:
        """Monitor active requests."""
        while True:
            active_count = len(self.active_requests)
            if active_count > 0:
                logger.info(f"Active requests: {active_count}")
            
            # Check timeouts
            current_time = datetime.now()
            for request_id, request in list(self.active_requests.items()):
                if (current_time - request.timestamp).total_seconds() > request.timeout:
                    await self.cancel_request(request_id)
                    logger.warning(f"Request {request_id} cancelled due to timeout")

            await asyncio.sleep(interval)