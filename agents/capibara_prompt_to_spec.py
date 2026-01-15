# capibara/agents/capibara_prompt_to_spec.py

class CapibaraPromptToAgentSpec:
    def __init__(self, model):
        self.model = model  # Can be CapibaraModel or any open source wrapper

    def generate_spec(self, instruction: str) -> dict:
        prompt = f"""
You are an agent generator for an AI system called CapibaraGPT.

Your task is to create a JSON specification with this structure:
{{
  "name": "...",           <- agent name
  "llm": {{
    "type": "ollama",      <- only use open source LLMs
    "model": "llama3"      <- local model
  }},
  "tools": ["..."],        <- list of tools to use (e.g.: "add", "web_search", etc.)
  "vectordb": {{
    "type": "qdrant"       <- if document context is needed, include this
  }}
}}

User instruction:
{instruction}

Return only the JSON with the specification.
        """

        raw = self.model.generate(prompt)
        try:
            import json
            return json.loads(raw)
        except Exception:
            raise ValueError(f"Error interpreting model output: {raw}")
