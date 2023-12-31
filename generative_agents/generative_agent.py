## Modified from Langchain library. You can refer the definitions of the GenerativeAgent and GenerativeAgentMemory 
## in the reference documentation for the following imports, focusing on add_memory and summarize_related_memories methods.
## Also: https://github.com/langchain-ai/langchain/blob/master/cookbook/generative_agents_interactive_simulacra_of_human_behavior.ipynb

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from generative_agents.memory import GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
# from langchain.schema.language_model import BaseLanguageModel


class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    """The character's name."""

    age: Optional[int] = None
    """The optional age of the character."""
    traits: str = "N/A"
    """Permanent traits to ascribe to the character."""
    status: str
    """The status of the character."""
    description: str = "N/A"
    """The backstory of the character revolves around what personality a character has."""
    memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    llm: BaseLanguageModel
    """The underlying language model."""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""

    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""

    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""

    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
"""
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        # q1 = f"What is the relationship between {self.name} and {entity_name}"
        q1 = f"What is the status of {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()
        return self.chain(prompt=prompt).run()

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M:%S %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M:%S %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

       
    def generate_reaction_for_route(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """React to a given observation."""
        call_to_action_template = (
            "{agent_name} should react to the observation, and if so,"
            + " what should be the next step for {agent_name} to move from the"
            + " given directions? Respond by only one direction {agent_name}"
            + " should focus on moving to. The potential responses should have one of the four"
            + " directions: forward, backward, left, and right. "
            +  "However, {agent_name} should see which "
            + " directions are available for him and most likely to have a tree-house based "
            + " on the observation and respond accordingly. Also the {agent_name} should justify why it wishes to go "
            + " in that particular direction and also why not the other direction based on his/her personality."
            +  "write \nREACT: {agent_name}'s reaction in the format: {agent_name} wish to go to .... direction because .. , "
            +  "{agent_name} does not wish to go to .... direction because . Be precise and write only the things that you know from the context. "
            +  "Double-check on which directions {agent_name} can move to."
            # +  "If the observation says restaurant, write 'destination found' after providing the directions as described above. "   "\n\n"
            +  "\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            extracted_direction = re.findall(r'\b(?:forward|left|right|backward)\b', reaction)  ###to extract the direction from the reaction

            if len(extracted_direction) > 0:

                return extracted_direction[0], f"{self.name} {reaction}"

            
            else:

                default_text = "No direction was provided in the reaction"

                return default_text, f"{self.name} {reaction}"
  
    def generate_perceived_ratings(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """React to a given observation."""
        call_to_action_template = (
            "{agent_name} should rate the observation."
            + " {agent_name} is currently given a task to judge the scenes on how safe they are according to the him/her."
            + " what rating from 1 (lower) to 10 (higher) it should give to the observed description of the scene."
            + " The ratings on safety depends on the {agent_name} personality and experiences. "
            + " The ratings also depends on the other previous ratings given by {agent_name}. "
            # + " It is must for the {agent_name} to observe carefully the details before providing output. "
            + " write \nREACT: {agent_name}'s reaction in the format: {agent_name} will give the scene ... out of 10."
            + " Elaborate on rating is based on my personality and experience."
            + " In {agent_name} opinion, the scene should have these properties ...... to make it a 10. "
            +  "\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            extracted_rating = re.findall(r'\b([1-9]|10)\b', reaction)  ###to extract the rating from the reaction

            print(extracted_rating, reaction)


            if int(extracted_rating[0]) <= 10:

                return extracted_rating[0], f"{self.name} {reaction}"
            
            else:

                default_text = "No rating was provided in the reaction"

                return default_text, f"{self.name} {reaction}"    
            

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def _compute_agent_summary(self) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name} (age: {age})"
            + f"\nInnate traits: {self.traits}"
            + f"\nBackstory and personality: {self.description}"
            + f"\n{self.summary}"
        )

    def get_full_header(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )
