"""Multi-LLM orchestration examples for u-llm-sdk."""
import asyncio
from u_llm_sdk import LLM
from u_llm_sdk.types import Provider
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.llm.providers import ClaudeProvider, CodexProvider, GeminiProvider
from u_llm_sdk.multi_llm import (
    ClarityGate,
    ConsensusLoop,
    BrainstormModule,
)
from u_llm_sdk.types.orchestration import Task, BrainstormConfig


async def clarity_gate_example():
    """Assess task clarity before execution."""
    config = LLMConfig(provider=Provider.CLAUDE)
    provider = ClaudeProvider(config)
    
    gate = ClarityGate(provider)
    task = Task(
        task_id="t1",
        objective="Build user authentication",
        context="FastAPI backend with PostgreSQL",
    )
    
    assessment = await gate.assess(task)
    print(f"Clarity Level: {assessment.level}")
    print(f"Recommendation: {assessment.recommendation}")
    
    if assessment.missing_info:
        print(f"Missing Info: {assessment.missing_info}")


async def consensus_loop_example():
    """Multi-provider voting for decisions."""
    # Create providers
    providers = {}
    
    # Check which providers are available
    if ClaudeProvider.is_available():
        providers[Provider.CLAUDE] = ClaudeProvider(LLMConfig(provider=Provider.CLAUDE))
    if GeminiProvider.is_available():
        providers[Provider.GEMINI] = GeminiProvider(LLMConfig(provider=Provider.GEMINI))
    if CodexProvider.is_available():
        providers[Provider.CODEX] = CodexProvider(LLMConfig(provider=Provider.CODEX))
    
    if len(providers) < 2:
        print("Need at least 2 providers for consensus voting")
        return
    
    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,  # 2/3 majority
        low_agreement_threshold=0.4,  # Below this → ask user
    )
    
    loop = ConsensusLoop(providers, config)
    result = await loop.run("Should we use microservices or monolith for this project?")
    
    if result.success:
        print(f"Consensus reached: {result.final_decision}")
    else:
        print(f"No consensus. User questions: {result.user_questions}")


async def brainstorm_example():
    """Full 3-round brainstorming session."""
    providers = {}
    
    if ClaudeProvider.is_available():
        providers[Provider.CLAUDE] = ClaudeProvider(LLMConfig(provider=Provider.CLAUDE))
    if GeminiProvider.is_available():
        providers[Provider.GEMINI] = GeminiProvider(LLMConfig(provider=Provider.GEMINI))
    
    if len(providers) < 2:
        print("Need at least 2 providers for brainstorming")
        return
    
    module = BrainstormModule(providers)
    result = await module.run_session("What's the best architecture for a real-time chat app?")
    
    print(f"Rounds completed: {len(result.rounds)}")
    print(f"Final consensus: {result.consensus}")
    
    # Full discussion is preserved without summarization
    for entry in result.all_discussion_entries[:5]:  # First 5 entries
        print(f"- {entry.provider.value}: {entry.content[:100]}...")


if __name__ == "__main__":
    print("=== Clarity Gate Example ===")
    asyncio.run(clarity_gate_example())
    
    print("\n=== Consensus Loop Example ===")
    asyncio.run(consensus_loop_example())
    
    print("\n=== Brainstorm Example ===")
    asyncio.run(brainstorm_example())
