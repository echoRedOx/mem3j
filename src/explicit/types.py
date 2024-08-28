from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List
from uuid import uuid4


@dataclass
class Message:
    """
    Standardized struct for messages.
    """
    uuid: str
    role: str
    speaker: str
    content: str
    timestamp: str = str(datetime.now().strftime('%Y-%m-%d @ %H:%M'))
    
    def to_dict(self):
        """
        Exports the Message container to an iterable

        :param self: The message instance to convert
        :return: dict representation
        """
        return asdict(self)

    def to_prompt_message_string(self):
        """
        Converts a Message object to a string that can be sent to the model.

        :param message: The Message object to convert.
        :return: A string that can be sent to the model.
        """
        
        return f"<|im_start|>{self.speaker} (Timestamp: {self.timestamp}): \n{self.content}<|im_end|>"
    
    def to_memory_string(self):
        """
        Converts a Message object to a string more suitable for context recall.

        :param message: The Message object to convert.
        :return: A string that can be sent to the model.
        """
        return f"{self.speaker} @ {self.timestamp}: {self.content}"


@dataclass
class Turn:
    """
    Class representing the Request/Response pairs of a conversation,
    
    :param request: Message type request object
    :param response: Message type response object
    :return: None
    """
    uuid: str
    request: Message
    response: Message

    def to_dict(self):
        """
        Converts the Turn dataclass instance to a dictionary.
        """
        return asdict(self)


@dataclass
class Conversation:
    uuid: str
    created_at: str
    last_active: str
    host: str
    host_is_bot: bool
    guest: str
    guest_is_bot: bool
    turns: List[Turn] = field(default_factory=list)

    def to_dict_dep(self):
        return {
            "uuid": self.uuid,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "host": self.host,
            "host_is_bot": self.host_is_bot,
            "guest": self.guest,
            "guest_is_bot": self.guest_is_bot,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    def to_dict(self):
        return asdict(self)
    
    def create_turn(self, request: Message, response: Message) -> Turn:
        """
        Creates a new MessageTurn object.

        :param request: The request Message object.
        :param response: The response Message object.

        :return: A MessageTurn object.
        """
        return Turn(
            uuid=str(uuid4()),
            request=request,
            response=response
        )


@dataclass
class MessageCache:
    """
    This class manages the conversation history for inclusion in prompt context injection as a deque with structural
    preservation on i/o
    """

    def __init__(self, capacity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity
        self.cache = deque(maxlen=capacity)

    def add_message(self, turn: Turn):
        self.cache.append(turn)

    def get_message_cache(self):
        message_cache = list(self.cache)
        return message_cache
    
    def get_n_messages(self, n):
        message_cache = list(self.cache)[-n:]
        return message_cache
    
    def get_chat_history(self):
        chat_history = []
        for turn in self.cache:
            chat_history.append(turn.request.to_prompt_message_string())
            chat_history.append(turn.response.to_prompt_message_string())
        return chat_history
