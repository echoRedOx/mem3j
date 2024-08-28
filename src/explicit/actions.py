from datetime import datetime
from uuid import uuid4
import yaml
from types import Conversation, Turn


def start_new_conversation(host: str, host_is_bot: bool, guest: str, guest_is_bot: bool) -> Conversation:
    """
    Starts a new conversation and returns the conversation UUID.

    :param host: The name of the host
    :param host_is_bot: Whether the host is a bot
    :param guest: The name of the guest iso
    :param guest_is_bot: Whether the guest is a bot

    :return: Conversation object
    """
    conversation = Conversation(
        uuid=str(uuid4()),
        created_at=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
        last_active=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
        host=host,
        host_is_bot=host_is_bot,
        guest=guest,
        guest_is_bot=guest_is_bot,
        turns=[]
    )
    
    print(f"New conversation {conversation.uuid} started!")

    return conversation


def append_turn_to_conversation_yaml(conversations_file_path: str, conversation_uuid: str, turn: Turn) -> None:
    """
    Appends a turn to the specified conversation in the YAML file.

    :param conversation_file_path: The file path of the YAML file.
    :param conversation_uuid: The UUID of the conversation to append to.
    :param turn: The turn data to append.
    """
    turn_dict = turn.to_dict()
    # Load the existing conversations
    with open(conversations_file_path, 'r') as file:
        data = yaml.safe_load(file) or {"conversations": []}

    # Find the conversation by UUID
    conversation = next((c for c in data["conversations"] if c["uuid"] == conversation_uuid), None)
    
    if conversation:
        conversation["turns"].append(turn_dict)
        conversation["last_active"] = turn_dict["response"]["timestamp"]
    else:
        # Create a new conversation if the uuid is not found.
        conversation = {
            "uuid": conversation_uuid,
            "created_at": turn_dict["request"]["timestamp"],
            "last_active": turn_dict["response"]["timestamp"],
            "turns": [turn_dict]
        }
        data["conversations"].append(conversation)
    
    # Write the updated list of conversations back to the YAML file
    with open(conversations_file_path, 'w') as file:
        yaml.safe_dump(data, file)
