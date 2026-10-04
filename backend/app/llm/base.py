from typing import Optional, Union, Dict, Any
from typing_extensions import Protocol

class LLMProvider(Protocol):
    async def is_available(self) -> bool:
        ...

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        think: bool = True,
        num_predict: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        ...

    @property
    def input_tokens(self) -> int:
        return 0

    @property
    def output_tokens(self) -> int:
        return 0

    @property
    def api_calls(self) -> int:
        return 0
