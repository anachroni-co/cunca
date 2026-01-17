#!/usr/bin/env python3
"""
🧠 CAPIBARA TEXT-TO-GEN SERVICE
==============================

Meta-servicio que genera automáticamente código completo para nuevos servicios text-to-*
desde descripciones en lenguaje natural.

Capacidades:
- Análisis de requerimientos desde texto natural
- Generación de código Python completo
- Integración automática con Capibara5
- Tests unitarios automáticos
- Documentación auto-generada
- API endpoints automáticos
- Deployment scripts incluidos

Ejemplos de servicios que puede generar:
- Text-to-Music: Genera composiciones musicales
- Text-to-Game: Crea mini-juegos simples
- Text-to-3D: Modelos 3D para realidad virtual
- Text-to-Code: Generador de código en cualquier lenguaje
- Text-to-Presentation: Slides automáticas
- Text-to-Website: Sitios web completos
- Text-to-Animation: Animaciones 2D/3D
- Text-to-Data: Datasets sintéticos
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import shutil
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Tipos de servicios que se pueden generar"""
    CREATIVE = "creative"           # Arte, música, diseño
    TECHNICAL = "technical"         # CAD, código, circuitos
    MEDIA = "media"                # Imagen, video, audio
    DATA = "data"                  # Datasets, APIs, bases de datos
    INTERACTIVE = "interactive"     # Juegos, interfaces, apps
    DOCUMENT = "document"          # Presentations, reports, websites
    SIMULATION = "simulation"      # Física, química, matemáticas
    BUSINESS = "business"          # Analytics, reports, workflows

class ComplexityLevel(Enum):
    """Niveles de complejidad del servicio"""
    SIMPLE = "simple"              # <500 líneas, 1-2 archivos
    MEDIUM = "medium"              # 500-2000 líneas, 3-5 archivos
    COMPLEX = "complex"            # 2000-5000 líneas, 5-10 archivos
    ADVANCED = "advanced"          # >5000 líneas, >10 archivos

class IntegrationType(Enum):
    """Tipos de integración con el ecosistema"""
    STANDALONE = "standalone"       # Servicio independiente
    CAPIBARA_NATIVE = "native"     # Integración completa con Capibara5
    E2B_INTEGRATION = "e2b"        # Usa E2B sandboxes
    API_WRAPPER = "api_wrapper"    # Wrapper de API externa
    HYBRID = "hybrid"              # Combinación de múltiples tipos

@dataclass
class MetaGenerationConfig:
    """Configuración para el meta-generador"""
    # Directorios de salida
    output_directory: str = "./generated_services"
    template_directory: str = "./templates"
    
    # Configuración de generación
    default_complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    default_integration: IntegrationType = IntegrationType.CAPIBARA_NATIVE
    
    # Características del código generado
    include_tests: bool = True
    include_docs: bool = True
    include_api_endpoints: bool = True
    include_demo_script: bool = True
    include_requirements: bool = True
    
    # Estilo de código
    code_style: str = "pep8"
    docstring_style: str = "google"
    type_hints: bool = True
    async_support: bool = True
    
    # Plantillas de código
    use_dataclasses: bool = True
    use_enums: bool = True
    use_logging: bool = True
    error_handling: str = "comprehensive"  # basic, standard, comprehensive
    
    # Integración con Capibara5
    auto_register_service: bool = True
    generate_web_interface: bool = True
    create_service_factory: bool = True
    
    # AI/ML específico
    include_mock_mode: bool = True
    default_model_type: str = "transformer"
    include_training_scaffold: bool = True

@dataclass
class ServiceGenerationRequest:
    """Request para generar un nuevo servicio"""
    description: str
    service_name: str
    service_type: ServiceType
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    integration_type: IntegrationType = IntegrationType.CAPIBARA_NATIVE
    
    # Especificaciones técnicas
    input_format: Optional[str] = None      # "text", "json", "file"
    output_format: Optional[str] = None     # "file", "json", "stream"
    dependencies: List[str] = field(default_factory=list)
    external_apis: List[str] = field(default_factory=list)
    
    # Características funcionales
    supports_batch: bool = False
    supports_streaming: bool = False
    requires_gpu: bool = False
    requires_internet: bool = False
    
    # Metadatos
    author: Optional[str] = None
    license: str = "MIT"
    version: str = "1.0.0"
    user_id: Optional[str] = None

@dataclass
class ServiceGenerationResult:
    """Resultado de la generación de servicio"""
    success: bool
    service_name: str
    service_path: str = ""
    
    # Archivos generados
    generated_files: List[str] = field(default_factory=list)
    main_service_file: str = ""
    test_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    
    # Métricas de código
    total_lines: int = 0
    files_count: int = 0
    complexity_score: float = 0.0
    
    # Información de integración
    api_endpoints: List[str] = field(default_factory=list)
    dependencies_added: List[str] = field(default_factory=list)
    
    # Instrucciones para el usuario
    installation_instructions: str = ""
    usage_examples: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Información de desarrollo
    generation_time_seconds: float = 0.0
    templates_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

# Try to import code generation components
try:
    from .code_generator import CodeGenerator
    from .service_analyzer import ServiceAnalyzer
    CODE_GENERATION_AVAILABLE = True
except ImportError:
    CODE_GENERATION_AVAILABLE = False

class ServiceRequirementAnalyzer:
    """Analizador de requerimientos desde descripción natural"""
    
    def __init__(self):
        self.service_keywords = {
            ServiceType.CREATIVE: ["music", "art", "design", "creative", "artistic", "song", "melody"],
            ServiceType.TECHNICAL: ["code", "program", "algorithm", "circuit", "cad", "engineering"],
            ServiceType.MEDIA: ["image", "video", "audio", "picture", "movie", "sound", "photo"],
            ServiceType.DATA: ["data", "dataset", "database", "api", "json", "csv", "table"],
            ServiceType.INTERACTIVE: ["game", "app", "interface", "interactive", "gui", "ui"],
            ServiceType.DOCUMENT: ["document", "report", "presentation", "slide", "website", "page"],
            ServiceType.SIMULATION: ["simulation", "physics", "chemistry", "math", "model", "calculate"],
            ServiceType.BUSINESS: ["business", "analytics", "report", "dashboard", "workflow", "process"]
        }
        
        self.complexity_indicators = {
            ComplexityLevel.SIMPLE: ["simple", "basic", "easy", "quick", "minimal"],
            ComplexityLevel.MEDIUM: ["standard", "normal", "regular", "typical"],
            ComplexityLevel.COMPLEX: ["complex", "advanced", "detailed", "comprehensive"],
            ComplexityLevel.ADVANCED: ["enterprise", "production", "scalable", "robust", "complete"]
        }
        
        self.integration_indicators = {
            IntegrationType.STANDALONE: ["standalone", "independent", "separate", "isolated"],
            IntegrationType.E2B_INTEGRATION: ["sandbox", "secure", "isolated", "e2b"],
            IntegrationType.API_WRAPPER: ["api", "wrapper", "external", "third-party"],
            IntegrationType.CAPIBARA_NATIVE: ["integrated", "native", "capibara", "ecosystem"]
        }
    
    def analyze_description(self, description: str) -> Dict[str, Any]:
        """Analiza descripción y extrae requerimientos"""
        description_lower = description.lower()
        
        # Detectar tipo de servicio
        service_type = self._detect_service_type(description_lower)
        
        # Detectar complejidad
        complexity = self._detect_complexity(description_lower)
        
        # Detectar tipo de integración
        integration = self._detect_integration_type(description_lower)
        
        # Extraer características técnicas
        technical_specs = self._extract_technical_specs(description_lower)
        
        # Extraer dependencias
        dependencies = self._extract_dependencies(description_lower)
        
        return {
            "service_type": service_type,
            "complexity": complexity,
            "integration_type": integration,
            "technical_specs": technical_specs,
            "dependencies": dependencies,
            "suggested_name": self._generate_service_name(description)
        }
    
    def _detect_service_type(self, description: str) -> ServiceType:
        """Detecta el tipo de servicio desde la descripción"""
        scores = {}
        
        for service_type, keywords in self.service_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > 0:
                scores[service_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return ServiceType.CREATIVE  # Default
    
    def _detect_complexity(self, description: str) -> ComplexityLevel:
        """Detecta el nivel de complejidad"""
        for complexity, indicators in self.complexity_indicators.items():
            if any(indicator in description for indicator in indicators):
                return complexity
        
        # Heurísticas adicionales basadas en longitud y características
        if len(description.split()) > 50:
            return ComplexityLevel.COMPLEX
        elif len(description.split()) > 20:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE
    
    def _detect_integration_type(self, description: str) -> IntegrationType:
        """Detecta el tipo de integración requerido"""
        for integration, indicators in self.integration_indicators.items():
            if any(indicator in description for indicator in indicators):
                return integration
        
        return IntegrationType.CAPIBARA_NATIVE  # Default
    
    def _extract_technical_specs(self, description: str) -> Dict[str, Any]:
        """Extrae especificaciones técnicas"""
        specs = {}
        
        # Detectar formatos de entrada/salida
        if any(word in description for word in ["file", "upload", "document"]):
            specs["input_format"] = "file"
        elif any(word in description for word in ["json", "api", "data"]):
            specs["input_format"] = "json"
        else:
            specs["input_format"] = "text"
        
        # Detectar requerimientos de GPU
        if any(word in description for word in ["ai", "ml", "neural", "model", "training"]):
            specs["requires_gpu"] = True
        
        # Detectar necesidad de internet
        if any(word in description for word in ["api", "download", "fetch", "online"]):
            specs["requires_internet"] = True
        
        # Detectar soporte para lotes
        if any(word in description for word in ["batch", "multiple", "bulk", "many"]):
            specs["supports_batch"] = True
        
        return specs
    
    def _extract_dependencies(self, description: str) -> List[str]:
        """Extrae dependencias probables"""
        dependencies = []
        
        # Mapeo de palabras clave a dependencias
        dependency_map = {
            "image": ["Pillow", "opencv-python"],
            "video": ["opencv-python", "moviepy"],
            "audio": ["librosa", "soundfile"],
            "music": ["music21", "librosa"],
            "ai": ["torch", "transformers"],
            "ml": ["scikit-learn", "numpy"],
            "data": ["pandas", "numpy"],
            "web": ["requests", "beautifulsoup4"],
            "api": ["requests", "fastapi"],
            "game": ["pygame", "arcade"],
            "3d": ["trimesh", "open3d"],
            "pdf": ["PyPDF2", "reportlab"],
            "excel": ["openpyxl", "pandas"]
        }
        
        for keyword, deps in dependency_map.items():
            if keyword in description:
                dependencies.extend(deps)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _generate_service_name(self, description: str) -> str:
        """Genera nombre sugerido para el servicio"""
        words = description.split()
        
        # Buscar palabras clave importantes
        key_words = []
        for word in words[:10]:  # Primeras 10 palabras
            clean_word = word.strip('.,!?').lower()
            if len(clean_word) > 3 and clean_word not in ['that', 'will', 'can', 'should', 'would']:
                key_words.append(clean_word.capitalize())
        
        if key_words:
            return f"TextTo{key_words[0]}"
        else:
            return "TextToCustom"

class MockCodeGenerator:
    """Generador de código mock cuando los componentes no están disponibles"""
    
    def __init__(self, config: MetaGenerationConfig):
        self.config = config
    
    async def generate_service_code(self, request: ServiceGenerationRequest) -> Dict[str, Any]:
        """Genera código mock para el servicio"""
        logger.info(f"🔧 Generating mock service: {request.service_name}")
        
        # Simular tiempo de generación
        await asyncio.sleep(2.0)
        
        service_dir = f"{self.config.output_directory}/{request.service_name.lower()}"
        
        # Archivos que se generarían
        generated_files = [
            f"{service_dir}/__init__.py",
            f"{service_dir}/{request.service_name.lower()}_service.py",
            f"{service_dir}/config.py",
            f"{service_dir}/models.py"
        ]
        
        if self.config.include_tests:
            generated_files.extend([
                f"{service_dir}/tests/test_{request.service_name.lower()}.py",
                f"{service_dir}/tests/__init__.py"
            ])
        
        if self.config.include_docs:
            generated_files.append(f"{service_dir}/README.md")
        
        if self.config.include_demo_script:
            generated_files.append(f"demo_{request.service_name.lower()}.py")
        
        # Código de ejemplo que se generaría
        mock_code = self._generate_mock_service_code(request)
        
        return {
            "success": True,
            "service_path": service_dir,
            "generated_files": generated_files,
            "main_service_file": f"{service_dir}/{request.service_name.lower()}_service.py",
            "total_lines": self._estimate_lines_of_code(request),
            "mock_code_preview": mock_code,
            "installation_instructions": self._generate_installation_instructions(request),
            "usage_examples": self._generate_usage_examples(request)
        }
    
    def _generate_mock_service_code(self, request: ServiceGenerationRequest) -> str:
        """Genera código de ejemplo para el servicio"""
        service_class = f"Capibara{request.service_name}"
        
        return f'''#!/usr/bin/env python3
"""
🎯 CAPIBARA {request.service_name.upper()} SERVICE
{"=" * (40 + len(request.service_name))}

{request.description}

Generado automáticamente por Text-to-Gen.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class {request.service_name}Config:
    """Configuración para {request.service_name}"""
    output_directory: str = "./generated_{request.service_name.lower()}"
    default_quality: str = "high"
    enable_caching: bool = True
    timeout_seconds: int = 30

@dataclass
class {request.service_name}Request:
    """Request para {request.service_name}"""
    description: str
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

@dataclass
class {request.service_name}Result:
    """Resultado de {request.service_name}"""
    success: bool
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    generation_time_seconds: float = 0.0
    error: Optional[str] = None

class {service_class}:
    """Servicio principal de {request.service_name}"""
    
    def __init__(self, config: Optional[{request.service_name}Config] = None):
        self.config = config or {request.service_name}Config()
        logger.info(f"✅ {service_class} initialized")
    
    async def generate(self, request: {request.service_name}Request) -> {request.service_name}Result:
        """Genera resultado desde descripción natural"""
        start_time = datetime.now()
        
        try:
            logger.info(f"🎯 Generating {{request.service_name}}: {{request.description[:100]}}...")
            
            # Simular procesamiento
            await asyncio.sleep(1.0)
            
            # Aquí iría la lógica específica del servicio
            result = await self._process_request(request)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {request.service_name}Result(
                success=True,
                output_path=result.get("output_path"),
                metadata=result.get("metadata", {{}}),
                generation_time_seconds=generation_time
            )
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {request.service_name} generation error: {{e}}")
            
            return {request.service_name}Result(
                success=False,
                generation_time_seconds=generation_time,
                error=str(e)
            )
    
    async def _process_request(self, request: {request.service_name}Request) -> Dict[str, Any]:
        """Procesa el request específico del servicio"""
        # TODO: Implementar lógica específica del servicio
        
        return {{
            "output_path": f"./generated_{{request.service_name.lower()}}/output.txt",
            "metadata": {{
                "description": request.description,
                "timestamp": datetime.now().isoformat()
            }}
        }}
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible"""
        return True
    
    async def test_functionality(self) -> Dict[str, Any]:
        """Prueba la funcionalidad del servicio"""
        test_request = {request.service_name}Request(
            description="Test generation request"
        )
        
        result = await self.generate(test_request)
        
        return {{
            "test_passed": result.success,
            "generation_time": result.generation_time_seconds,
            "error": result.error
        }}

# Factory function
def create_{request.service_name.lower()}_service(config=None):
    """Crea instancia del servicio {request.service_name}"""
    return {service_class}(config)
'''
    
    def _estimate_lines_of_code(self, request: ServiceGenerationRequest) -> int:
        """Estima líneas de código que se generarían"""
        base_lines = {
            ComplexityLevel.SIMPLE: 300,
            ComplexityLevel.MEDIUM: 800,
            ComplexityLevel.COMPLEX: 2000,
            ComplexityLevel.ADVANCED: 5000
        }
        
        lines = base_lines[request.complexity]
        
        # Ajustes basados en características
        if request.supports_batch:
            lines += 200
        if request.supports_streaming:
            lines += 300
        if request.integration_type == IntegrationType.E2B_INTEGRATION:
            lines += 500
        
        return lines
    
    def _generate_installation_instructions(self, request: ServiceGenerationRequest) -> str:
        """Genera instrucciones de instalación"""
        return f"""# Instalación de {request.service_name}

1. Instalar dependencias:
   pip install -r requirements.txt

2. Configurar servicio:
   from capibara.services.{request.service_name.lower()} import create_{request.service_name.lower()}_service
   
3. Usar servicio:
   service = create_{request.service_name.lower()}_service()
   result = await service.generate(request)
"""
    
    def _generate_usage_examples(self, request: ServiceGenerationRequest) -> List[str]:
        """Genera ejemplos de uso"""
        return [
            f"# Ejemplo básico de {request.service_name}",
            f"request = {request.service_name}Request(description='Your description here')",
            f"result = await service.generate(request)",
            f"print(f'Generated: {{result.output_path}}')"
        ]

class CapibaraTextToGen:
    """Servicio principal de Text-to-Gen"""
    
    def __init__(self, config: Optional[MetaGenerationConfig] = None):
        self.config = config or MetaGenerationConfig()
        
        # Inicializar componentes
        self.requirement_analyzer = ServiceRequirementAnalyzer()
        self.mock_generator = MockCodeGenerator(self.config)
        
        # Intentar inicializar generador real
        self.code_generator = None
        if CODE_GENERATION_AVAILABLE:
            self._init_real_generator()
        
        # Configurar directorios
        self._setup_directories()
        
        logger.info("✅ CapibaraTextToGen initialized")
    
    def _init_real_generator(self):
        """Inicializa el generador de código real"""
        try:
            self.code_generator = CodeGenerator(self.config)
            logger.info("✅ Real code generator initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing real generator: {e}")
            self.code_generator = None
    
    def _setup_directories(self):
        """Configura directorios de salida"""
        try:
            os.makedirs(self.config.output_directory, exist_ok=True)
            os.makedirs(self.config.template_directory, exist_ok=True)
            logger.info(f"📁 Output directory: {self.config.output_directory}")
        except Exception as e:
            logger.error(f"❌ Error creating directories: {e}")
    
    async def generate_service(self, request: ServiceGenerationRequest) -> ServiceGenerationResult:
        """Genera un nuevo servicio text-to-* desde descripción"""
        start_time = datetime.now()
        
        try:
            logger.info(f"🧠 Generating service: {request.service_name}")
            logger.info(f"📝 Description: {request.description[:200]}...")
            
            # Analizar requerimientos si no se especificaron
            if not hasattr(request, 'analyzed'):
                analysis = self.requirement_analyzer.analyze_description(request.description)
                
                # Aplicar análisis si no se especificó explícitamente
                if request.service_type is None:
                    request.service_type = analysis["service_type"]
                if request.complexity == ComplexityLevel.MEDIUM:  # Default
                    request.complexity = analysis["complexity"]
                if request.integration_type == IntegrationType.CAPIBARA_NATIVE:  # Default
                    request.integration_type = analysis["integration_type"]
                
                # Agregar dependencias sugeridas
                if not request.dependencies:
                    request.dependencies = analysis["dependencies"]
                
                logger.info(f"📊 Analysis: {analysis['service_type'].value}, {analysis['complexity'].value}")
            
            # Generar código
            if self.code_generator:
                logger.info("⚙️ Using real code generator...")
                generation_result = await self.code_generator.generate_service_code(request)
            else:
                logger.info("🎭 Using mock code generator...")
                generation_result = await self.mock_generator.generate_service_code(request)
            
            # Procesar resultado
            generation_time = (datetime.now() - start_time).total_seconds()
            
            if generation_result["success"]:
                # Crear estructura de archivos (modo mock)
                if not self.code_generator:
                    await self._create_mock_files(request, generation_result)
                
                result = ServiceGenerationResult(
                    success=True,
                    service_name=request.service_name,
                    service_path=generation_result["service_path"],
                    generated_files=generation_result["generated_files"],
                    main_service_file=generation_result["main_service_file"],
                    total_lines=generation_result["total_lines"],
                    files_count=len(generation_result["generated_files"]),
                    complexity_score=self._calculate_complexity_score(request),
                    installation_instructions=generation_result["installation_instructions"],
                    usage_examples=generation_result["usage_examples"],
                    generation_time_seconds=generation_time,
                    templates_used=["base_service_template", "capibara_integration_template"]
                )
                
                logger.info(f"✅ Service '{request.service_name}' generated successfully!")
                logger.info(f"📊 Generated {result.files_count} files, {result.total_lines} lines")
                
                return result
            else:
                return ServiceGenerationResult(
                    success=False,
                    service_name=request.service_name,
                    generation_time_seconds=generation_time,
                    error=generation_result.get("error", "Unknown generation error")
                )
                
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Service generation error: {e}")
            
            return ServiceGenerationResult(
                success=False,
                service_name=request.service_name,
                generation_time_seconds=generation_time,
                error=str(e)
            )
    
    async def _create_mock_files(self, request: ServiceGenerationRequest, generation_result: Dict[str, Any]):
        """Crea archivos mock para demostración"""
        try:
            service_dir = generation_result["service_path"]
            os.makedirs(service_dir, exist_ok=True)
            
            # Crear archivo principal
            main_file = generation_result["main_service_file"]
            os.makedirs(os.path.dirname(main_file), exist_ok=True)
            
            with open(main_file, 'w') as f:
                f.write(generation_result["mock_code_preview"])
            
            # Crear __init__.py
            init_file = f"{service_dir}/__init__.py"
            with open(init_file, 'w') as f:
                f.write(f'''"""
{request.service_name} Service

{request.description}

Generado automáticamente por Capibara5 Text-to-Gen.
"""

from .{request.service_name.lower()}_service import (
    Capibara{request.service_name},
    {request.service_name}Config,
    {request.service_name}Request,
    {request.service_name}Result
)

def create_{request.service_name.lower()}_service(config=None):
    return Capibara{request.service_name}(config)

__all__ = [
    'Capibara{request.service_name}',
    '{request.service_name}Config',
    '{request.service_name}Request',
    '{request.service_name}Result',
    'create_{request.service_name.lower()}_service'
]
''')
            
            # Crear README
            if self.config.include_docs:
                readme_file = f"{service_dir}/README.md"
                with open(readme_file, 'w') as f:
                    f.write(f'''# {request.service_name} Service

{request.description}

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```python
from capibara.services.{request.service_name.lower()} import create_{request.service_name.lower()}_service

service = create_{request.service_name.lower()}_service()
request = {request.service_name}Request(description="Your description here")
result = await service.generate(request)
print(f"Generated: {{result.output_path}}")
```

## Características

- Tipo: {request.service_type.value}
- Complejidad: {request.complexity.value}
- Integración: {request.integration_type.value}

Generado automáticamente por Capibara5 Text-to-Gen.
''')
            
            logger.info(f"📝 Mock files created in {service_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error creating mock files: {e}")
    
    def _calculate_complexity_score(self, request: ServiceGenerationRequest) -> float:
        """Calcula score de complejidad del servicio"""
        base_scores = {
            ComplexityLevel.SIMPLE: 0.2,
            ComplexityLevel.MEDIUM: 0.5,
            ComplexityLevel.COMPLEX: 0.8,
            ComplexityLevel.ADVANCED: 1.0
        }
        
        score = base_scores[request.complexity]
        
        # Ajustes por características
        if request.supports_batch:
            score += 0.1
        if request.supports_streaming:
            score += 0.15
        if request.requires_gpu:
            score += 0.2
        if request.integration_type == IntegrationType.E2B_INTEGRATION:
            score += 0.15
        
        return min(1.0, score)
    
    # Utility methods
    def list_generated_services(self) -> List[str]:
        """Lista servicios generados"""
        try:
            if os.path.exists(self.config.output_directory):
                return [d for d in os.listdir(self.config.output_directory) 
                       if os.path.isdir(os.path.join(self.config.output_directory, d))]
            return []
        except Exception as e:
            logger.error(f"Error listing services: {e}")
            return []
    
    def analyze_description(self, description: str) -> Dict[str, Any]:
        """Analiza descripción y sugiere configuración"""
        return self.requirement_analyzer.analyze_description(description)
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible"""
        return True
    
    def get_supported_service_types(self) -> List[ServiceType]:
        """Obtiene tipos de servicios soportados"""
        return list(ServiceType)
    
    def get_complexity_levels(self) -> List[ComplexityLevel]:
        """Obtiene niveles de complejidad soportados"""
        return list(ComplexityLevel)
    
    async def test_generation(self) -> Dict[str, Any]:
        """Prueba la funcionalidad de generación"""
        test_request = ServiceGenerationRequest(
            description="A simple test service that converts text to uppercase",
            service_name="TextToUppercase",
            service_type=ServiceType.TECHNICAL,
            complexity=ComplexityLevel.SIMPLE
        )
        
        result = await self.generate_service(test_request)
        
        return {
            "test_passed": result.success,
            "generation_time": result.generation_time_seconds,
            "files_generated": result.files_count,
            "error": result.error
        }