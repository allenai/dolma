import os
from dataclasses import dataclass, field

import aiohttp


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass(frozen=True)
class BaseApiRequest:
    endpoint: str
    messages: list[Message]
    parameters: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)

    async def make(self):
        payload = {**self.parameters, "messages": [message.to_dict() for message in self.messages]}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=self.headers) as response:
                return await response.json()


@dataclass(frozen=True)
class Gpt4oRequest(BaseApiRequest):
    model: str = "gpt-4o"
    temperature: float = 1.0
    top_p: float = 1.0
    headers: dict = field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
    )

    def __post_init__(self):
        self.parameters.update({
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p
        })
