"""Coach-guided random agent for ablations."""

from __future__ import annotations

from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.ideas import IdeaStore
from llm.memory import JsonlMemoryStore
from agents.hybrid import HybridAgent
from flappy.extract import DocumentExtractor
from flappy.memory import NoteStore
from flappy.rag import SimpleRAG


class CoachRandomAgent(HybridAgent):
    """Uses coach guidance but samples uniformly within masks."""

    def __init__(
        self,
        env: BrowserGymEnvWrapper,
        coach: Coach,
        *,
        memory: JsonlMemoryStore | None = None,
        planner_interval: int = 10,
        max_steps: int = 200,
        reflexion_enabled: bool = True,
        reflexion_read_only: bool = False,
        note_store: NoteStore | None = None,
        rag: SimpleRAG | None = None,
        extractor: DocumentExtractor | None = None,
        idea_store: IdeaStore | None = None,
        ddl_inject: bool = False,
        ddl_top_k: int = 3,
    ) -> None:
        super().__init__(
            env=env,
            coach=coach,
            learner=None,
            memory=memory,
            planner_interval=planner_interval,
            max_steps=max_steps,
            reflexion_enabled=reflexion_enabled,
            reflexion_read_only=reflexion_read_only,
            note_store=note_store,
            rag=rag,
            extractor=extractor,
            idea_store=idea_store,
            ddl_inject=ddl_inject,
            ddl_top_k=ddl_top_k,
        )
