# services

This folder contains optional, experimental services around CapibaraGPT v3. The goal here is to be honest
about what is implemented today and what is still scaffolded.

## What exists today

- `tts/` (real, optional): ONNX-based TTS pipeline with FastSpeech + HiFi-GAN and a WebSocket server.
- `automation/` (experimental): FastAPI service with workflow builder + agent/sandbox orchestration. Some
  execution paths are simulated when external systems are not available.
- `mcp_integration.py` (local only): in-process MCP message bus for testing; no network transport.
- `bim_generation/`, `genomic_analysis/`, `meta_generation/`, `print3d_generation/`: standalone prototypes
  not wired into the core training/inference loop.

## Prototype status (needs deep review)

- `bim_generation/`: uses mock generator and imports modules that do not exist yet.
- `genomic_analysis/`: uses mock analyzer and imports modules that do not exist yet.
- `meta_generation/`: mock code generation and simulated orchestration; external integrations not wired.
- `print3d_generation/`: mock generator used when E2B integrations are missing.

## TTS quick start (optional)

Requirements:
- `onnxruntime`, `websockets`, `pyttsx3`, `numpy`, `python-dotenv`
- ONNX models for FastSpeech and HiFi-GAN

Example:

```python
from capibara.services.tts import CapibaraTextToSpeech

# You can also rely on env vars: FASTSPEECH_MODEL_PATH, HIFIGAN_MODEL_PATH
tts = CapibaraTextToSpeech(
    fastspeech_model_path="models/fastspeech.onnx",
    hifigan_model_path="models/hifigan.onnx",
    sample_rate=22050,
)

# Generate audio via the fallback synthesizer
wav_bytes = tts.generate_audio("Hola, soy Capibara.")
```

Start the WebSocket server:

```python
tts.start_websocket_server(host="localhost", port=8765)
```

Notes:
- The TTS server is blocking; run it in its own process.
- If dependencies are missing, importing `capibara.services.tts` will raise a clear error.

## Automation service (experimental)

The automation service can expose a FastAPI app and optionally talk to n8n and E2B. It requires:

- `fastapi`, `uvicorn`, `aiohttp`, `pydantic`

Some execution paths are simulated when external services are not present. Treat this module as a
prototype that you can extend or replace for production use.

## MCP integration (local only)

`services/mcp_integration.py` provides a lightweight, in-process message bus. It is useful for local
coordination tests but does not ship with a network transport or persistence.

## Status disclaimer

These services are not part of the core training/inference pipeline unless you wire them in explicitly.
They are meant to be optional building blocks.

## Issues por hacer

- [ ] # Use standard n8n execution (simulated) - `services\automation\agent_executor.py:340`
- [ ] """Execute a node in standard n8n mode (simulated).""" - `services\automation\agent_executor.py:481`
- [ ] # Simulate different node behaviors - `services\automation\agent_executor.py:494`
- [ ] result["response"] = {"status": 200, "data": "simulated response"} - `services\automation\agent_executor.py:498`
- [ ] result["output"] = f"Simulated output for {node.type}" - `services\automation\agent_executor.py:502`
- [ ] Some execution paths are simulated when external services are unavailable. - `services\automation\n8n_service.py:5`
- [ ] # For now, simulate execution - `services\automation\n8n_service.py:576`
- [ ] # Simulate processing time - `services\automation\n8n_service.py:581`
- [ ] # For now, show a placeholder - `services\automation\web_ui.py:242`
- [ ] placeholder="Example: Send me an email when someone submits the contact form" required></textarea> - `services\automation\web_ui.py:325`
- [ ] class MockBIMGenerator: - `services\bim_generation\text_to_bim_service.py:309`
- [ ] """Generador BIM mock para demonstration""" - `services\bim_generation\text_to_bim_service.py:310`
- [ ] logger.info("️ Mock BIM Generator initialized") - `services\bim_generation\text_to_bim_service.py:313`
- [ ] """Genera modelo BIM mock""" - `services\bim_generation\text_to_bim_service.py:316`
- [ ] self.generator = MockBIMGenerator()  # Default to mock for demo - `services\bim_generation\text_to_bim_service.py:385`
- [ ] bim_tool_used="mock_generator", - `services\bim_generation\text_to_bim_service.py:435`
- [ ] # methods de utilidad - `services\bim_generation\text_to_bim_service.py:494`
- [ ] MockBIMGenerator - `services\bim_generation\__init__.py:66`
- [ ] "mock_bim": True,  # Siempre disponible para testing - `services\bim_generation\__init__.py:136`
- [ ] class MockGenomicAnalyzer: - `services\genomic_analysis\text_to_genomic_service.py:327`
- [ ] """Analizador genomic mock para demonstration (mientras se integran herramientas reales)""" - `services\genomic_analysis\text_to_genomic_service.py:328`
- [ ] logger.info(" Mock Genomic Analyzer initialized (60B model simulation)") - `services\genomic_analysis\text_to_genomic_service.py:331`
- [ ] mock_variants = [ - `services\genomic_analysis\text_to_genomic_service.py:337`
- [ ] "variants_found": len(mock_variants), - `services\genomic_analysis\text_to_genomic_service.py:363`
- [ ] "variants": mock_variants, - `services\genomic_analysis\text_to_genomic_service.py:364`
- [ ] self.analyzer = MockGenomicAnalyzer() - `services\genomic_analysis\text_to_genomic_service.py:468`
- [ ] # methods de utilidad - `services\genomic_analysis\text_to_genomic_service.py:666`
- [ ] # Servicios disponibles (mocks para demonstration) - `services\meta_generation\multi_service_orchestrator.py:297`
- [ ] "print3d": self._mock_print3d_service, - `services\meta_generation\multi_service_orchestrator.py:299`
- [ ] "circuit": self._mock_circuit_service, - `services\meta_generation\multi_service_orchestrator.py:300`
- [ ] "code": self._mock_code_service, - `services\meta_generation\multi_service_orchestrator.py:301`
- [ ] "cad": self._mock_cad_service, - `services\meta_generation\multi_service_orchestrator.py:302`
- [ ] "api": self._mock_api_service - `services\meta_generation\multi_service_orchestrator.py:303`
- [ ] # Esperar a que terminen todos los paralelos - `services\meta_generation\multi_service_orchestrator.py:405`
- [ ] # Mock services para demonstration - `services\meta_generation\multi_service_orchestrator.py:449`
- [ ] async def _mock_print3d_service(self, params: Dict[str, Any]) -> Dict[str, Any]: - `services\meta_generation\multi_service_orchestrator.py:450`
- [ ] """Mock del servicio Text-to-Print3D""" - `services\meta_generation\multi_service_orchestrator.py:451`
- [ ] async def _mock_circuit_service(self, params: Dict[str, Any]) -> Dict[str, Any]: - `services\meta_generation\multi_service_orchestrator.py:476`
- [ ] """Mock del servicio Text-to-Circuit""" - `services\meta_generation\multi_service_orchestrator.py:477`
- [ ] async def _mock_code_service(self, params: Dict[str, Any]) -> Dict[str, Any]: - `services\meta_generation\multi_service_orchestrator.py:504`
- [ ] """Mock del servicio Text-to-Code (firmware)""" - `services\meta_generation\multi_service_orchestrator.py:505`
- [ ] async def _mock_cad_service(self, params: Dict[str, Any]) -> Dict[str, Any]: - `services\meta_generation\multi_service_orchestrator.py:532`
- [ ] """Mock del servicio Text-to-CAD (ensamblaje)""" - `services\meta_generation\multi_service_orchestrator.py:533`
- [ ] async def _mock_api_service(self, params: Dict[str, Any]) -> Dict[str, Any]: - `services\meta_generation\multi_service_orchestrator.py:557`
- [ ] """Mock del servicio Text-to-API (control interface)""" - `services\meta_generation\multi_service_orchestrator.py:558`
- [ ] - Verificar que todos los componentes estén disponibles - `services\meta_generation\multi_service_orchestrator.py:723`
- [ ] """methods de shipping""" - `services\meta_generation\payment_and_fulfillment.py:64`
- [ ] supplier_payment.stripe_transfer_id = f"tr_mock_{order.order_id}_{supplier_payment.supplier_id}" - `services\meta_generation\payment_and_fulfillment.py:275`
- [ ] # Costes de shipping por method y región - `services\meta_generation\payment_and_fulfillment.py:319`
- [ ] include_mock_mode: bool = True - `services\meta_generation\text_to_gen_service.py:108`
- [ ] class MockCodeGenerator: - `services\meta_generation\text_to_gen_service.py:341`
- [ ] """Mock code generator when components are not available""" - `services\meta_generation\text_to_gen_service.py:342`
- [ ] """Generates mock code for the service""" - `services\meta_generation\text_to_gen_service.py:348`
- [ ] logger.info(f" Generating mock service: {request.service_name}") - `services\meta_generation\text_to_gen_service.py:349`
- [ ] # Simulate generation time - `services\meta_generation\text_to_gen_service.py:351`
- [ ] mock_code = self._generate_mock_service_code(request) - `services\meta_generation\text_to_gen_service.py:377`
- [ ] "mock_code_preview": mock_code, - `services\meta_generation\text_to_gen_service.py:385`
- [ ] def _generate_mock_service_code(self, request: ServiceGenerationRequest) -> str: - `services\meta_generation\text_to_gen_service.py:390`
- [ ] # Simulate processing - `services\meta_generation\text_to_gen_service.py:455`
- [ ] self.mock_generator = MockCodeGenerator(self.config) - `services\meta_generation\text_to_gen_service.py:628`
- [ ] logger.info(" Using mock code generator...") - `services\meta_generation\text_to_gen_service.py:689`
- [ ] generation_result = await self.mock_generator.generate_service_code(request) - `services\meta_generation\text_to_gen_service.py:690`
- [ ] # Create file structure (mock mode) - `services\meta_generation\text_to_gen_service.py:696`
- [ ] await self._create_mock_files(request, generation_result) - `services\meta_generation\text_to_gen_service.py:698`
- [ ] async def _create_mock_files(self, request: ServiceGenerationRequest, generation_result: Dict[str, Any]): - `services\meta_generation\text_to_gen_service.py:738`
- [ ] """Creates mock files for demonstration""" - `services\meta_generation\text_to_gen_service.py:739`
- [ ] f.write(generation_result["mock_code_preview"]) - `services\meta_generation\text_to_gen_service.py:749`
- [ ] logger.info(f" Mock files created in {service_dir}") - `services\meta_generation\text_to_gen_service.py:815`
- [ ] logger.error(f" Error creating mock files: {e}") - `services\meta_generation\text_to_gen_service.py:818`
- [ ] class MockPrint3DGenerator: - `services\print3d_generation\text_to_print3d_service.py:367`
- [ ] """Mock generator for Print3D models when E2B is not available.""" - `services\print3d_generation\text_to_print3d_service.py:368`
- [ ] async def generate_print3d_mock(self, request: Print3DRequest) -> Print3DResult: - `services\print3d_generation\text_to_print3d_service.py:373`
- [ ] """Generate a mock 3D model optimized for printing.""" - `services\print3d_generation\text_to_print3d_service.py:374`
- [ ] logger.info(f"️ Generating mock Print3D model: {request.description[:100]}...") - `services\print3d_generation\text_to_print3d_service.py:375`
- [ ] # Simulate generation time - `services\print3d_generation\text_to_print3d_service.py:377`
- [ ] # Simulate print analysis - `services\print3d_generation\text_to_print3d_service.py:385`
- [ ] volume = self._calculate_mock_volume(request) - `services\print3d_generation\text_to_print3d_service.py:386`
- [ ] tool_used="mock_generator" - `services\print3d_generation\text_to_print3d_service.py:434`
- [ ] def _calculate_mock_volume(self, request: Print3DRequest) -> float: - `services\print3d_generation\text_to_print3d_service.py:437`
- [ ] # Initialize parser and mock generator - `services\print3d_generation\text_to_print3d_service.py:492`
- [ ] self.mock_generator = MockPrint3DGenerator(self.config) - `services\print3d_generation\text_to_print3d_service.py:494`
- [ ] # Fallback to mock generation - `services\print3d_generation\text_to_print3d_service.py:639`
- [ ] logger.info(" Using mock Print3D generation (E2B tools not available)") - `services\print3d_generation\text_to_print3d_service.py:640`
- [ ] result = await self.mock_generator.generate_print3d_mock(request) - `services\print3d_generation\text_to_print3d_service.py:641`
- [ ] results["mock_generator"] = {"available": True, "status": "Always available"} - `services\print3d_generation\text_to_print3d_service.py:767`
- [ ] """Placeholder that raises if TTS dependencies are missing.""" - `services\tts\__init__.py:12`
- [ ] raise RuntimeError("TTS is unavailable because dependencies are missing.") - `services\tts\__init__.py:21`
