import uuid
from typing import List, Optional

from .utils import logging

logger = logging.get_logger(__name__)


class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the
    addition of new user input and generated model responses. A conversation needs to contain an unprocessed user input
    before being passed to the :class:`~transformers.ConversationalPipeline`. This user input is either created when
    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response("input")` after a
    conversation turn.
    Arguments:
        text (:obj:`str`, `optional`):
            The initial user input to start the conversation. If not provided, a user input needs to be provided
            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can
            begin.
        conversation_id (:obj:`uuid.UUID`, `optional`):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.
        past_user_inputs (:obj:`List[str]`, `optional`):
            Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and
            :obj:`generated_responses` with equal length lists of strings
        generated_responses (:obj:`List[str]`, `optional`):
            Eventual past history of the conversation of the model. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and
            :obj:`generated_responses` with equal length lists of strings
    Usage::
        conversation = Conversation("Going to the movies tonight - any suggestions?")
        # Steps usually performed by the model when generating a response:
        # 1. Mark the user input as processed (moved to the history)
        conversation.mark_processed()
        # 2. Append a mode response
        conversation.append_response("The Big lebowski.")
        conversation.add_user_input("Is it good?")
    """
    
    def __init__(
            self, text: str = None, conversation_id: uuid.UUID = None, past_user_inputs=None, generated_responses=None
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        if past_user_inputs is None:
            past_user_inputs = []
        if generated_responses is None:
            generated_responses = []
        
        self.uuid: uuid.UUID = conversation_id
        self.past_user_inputs: List[str] = past_user_inputs
        self.generated_responses: List[str] = generated_responses
        self.new_user_input: Optional[str] = text
    
    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        if self.uuid == other.uuid:
            return True
        return (
                self.new_user_input == other.new_user_input
                and self.past_user_inputs == other.past_user_inputs
                and self.generated_responses == other.generated_responses
        )
    
    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
        field.
        Args:
            text (:obj:`str`): The user input for the next conversation round.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not existing and unprocessed user input should be overwritten when this function is called.
        """
        if self.new_user_input:
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" was overwritten '
                    f'with: "{text}".'
                )
                self.new_user_input = text
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
                )
        else:
            self.new_user_input = text
    
    def mark_processed(self):
        """
        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
        empties the :obj:`new_user_input` field.
        """
        if self.new_user_input:
            self.past_user_inputs.append(self.new_user_input)
        self.new_user_input = None
    
    def append_response(self, response: str):
        """
        Append a response to the list of generated responses.
        Args:
            response (:obj:`str`): The model generated response.
        """
        self.generated_responses.append(response)
    
    def iter_texts(self):
        """
        Iterates over all blobs of the conversation.
        Returns: Iterator of (is_user, text_chunk) in chronological order of the conversation. ``is_user`` is a
        :obj:`bool`, ``text_chunks`` is a :obj:`str`.
        """
        for user_input, generated_response in zip(self.past_user_inputs, self.generated_responses):
            yield True, user_input
            yield False, generated_response
        if self.new_user_input:
            yield True, self.new_user_input
    
    def __repr__(self):
        """
        Generates a string representation of the conversation.
        Return:
            :obj:`str`:
            Example: Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user >> Going to the movies tonight - any
            suggestions? bot >> The Big Lebowski
        """
        output = f"Conversation id: {self.uuid} \n"
        for is_user, text in self.iter_texts():
            name = "user" if is_user else "bot"
            output += f"{name} >> {text} \n"
        return output
