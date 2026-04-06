"""
AI-AI baseline simulation.

Generate simulated conversations for null hypothesis testing using multiple LLM providers.
Supports OpenAI, Anthropic, and HuggingFace (OpenAI-compatible).

This simulation matches the actual PENPAL human-AI experiment configuration:
- Temperature: 1.0
- Max tokens: 35
- System prompt: Collaborative storytelling prompt with pacing info
"""

from typing import Optional, List, Dict, Literal
from abc import ABC, abstractmethod
import pandas as pd
import os
import time
import random
from tqdm import tqdm


# Provider type definitions
ProviderType = Literal["openai", "anthropic", "huggingface"]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 1.0,
                 max_tokens: int = 35) -> str:
        """Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (default 1.0 to match experiment)
            max_tokens: Maximum output tokens (default 35 to match experiment)
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def reset_conversation(self):
        """Reset conversation state for a new story."""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with conversation threading."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.conversation_id = None
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 1.0,
                 max_tokens: int = 35) -> str:
        """Generate using OpenAI's responses API with conversation threading."""
        # Build input from last user message
        last_msg = messages[-1]['content'] if messages else ""
        
        if self.conversation_id is None:
            # Create new conversation with system prompt
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
            
            conversation = self.client.conversations.create(
                metadata={"topic": "ai-ai-simulation"},
                items=[
                    {"type": "message", "role": "developer", "content": system_msg},
                    {"type": "message", "role": "user", "content": ""}
                ]
            )
            self.conversation_id = conversation.id
        
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conversation_id,
            input=last_msg,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Extract text from response
        out = ""
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    if isinstance(item.content, str):
                        out += item.content
                    elif isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                out += content_item.text
        
        return out.strip()
    
    def reset_conversation(self):
        """Reset conversation ID for new story."""
        self.conversation_id = None


class AnthropicProvider(BaseProvider):
    """Anthropic API provider using messages API."""
    
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.conversation_history = []
        self.system_prompt = ""
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 1.0,
                 max_tokens: int = 35) -> str:
        """Generate using Anthropic's messages API with rolling history."""
        # Extract system prompt (keep separate for Anthropic)
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        if system_msg:
            self.system_prompt = system_msg
        
        # Add user messages to history (skip system)
        user_msgs = [m for m in messages if m['role'] != 'system']
        if user_msgs:
            last_msg = user_msgs[-1]
            self.conversation_history.append({
                "role": last_msg['role'],
                "content": last_msg['content']
            })
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_prompt,
            messages=self.conversation_history
        )
        
        # Extract response text
        out = response.content[0].text if response.content else ""
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": out
        })
        
        return out.strip()
    
    def reset_conversation(self):
        """Reset conversation history for new story."""
        self.conversation_history = []
        self.system_prompt = ""


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider (OpenAI-compatible).
    
    Uses HuggingFace's OpenAI-compatible endpoint, matching the original 
    PENPAL experiment setup which uses the OpenAI SDK with a custom base_url.
    """
    
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://router.huggingface.co/v1"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.conversation_history = []
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 1.0,
                 max_tokens: int = 35) -> str:
        """Generate using HuggingFace's OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""
    
    def reset_conversation(self):
        """Reset conversation history for new story."""
        self.conversation_history = []


def get_provider(provider_type: ProviderType, api_key: str, model_name: str, 
                 base_url: str = None) -> BaseProvider:
    """Factory function to create provider instances.
    
    Args:
        provider_type: One of 'openai', 'anthropic', 'huggingface'
        api_key: API key for the provider
        model_name: Model name/identifier
        base_url: Base URL for HuggingFace (OpenAI-compatible endpoint)
        
    Returns:
        Provider instance
    """
    if provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model_name=model_name)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(api_key=api_key, model_name=model_name, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


class SystemPrompts:
    """System prompts matching the actual PENPAL human-AI experiment.
    
    IMPORTANT: Both agents get the IDENTICAL prompt - the collaborative storytelling
    instruction. The only difference is turn order. This matches the actual experiment
    where both human and AI see the same instructions.
    """
    
    # Base prompt from the actual experiment (AppContext.jsx)
    # This is the SAME prompt given to BOTH agents
    STORY_COLLABORATOR_PROMPT = """You are an author taking part in a collaborative storytelling game activity with another author.
Together, you will create a story by taking turns adding to it. 
Your goal is to continue from where your partner has left off.
If there's no story, please begin the story. You have 10 interactions to write the story. Your input may get slightly truncated with a random character amount."""

    # The starting context that the first agent continues from
    STORY_PREFIX = "This is the story of"

    @staticmethod
    def get_system_prompt(turn_number: int = 1) -> str:
        """Get system prompt for any agent (both agents get identical prompts).
        
        Args:
            turn_number: Current turn number for pacing info
            
        Returns:
            System prompt with pacing metadata
        """
        base = SystemPrompts.STORY_COLLABORATOR_PROMPT
        # Add pacing info like the real experiment server does
        pacing = f"""

[Session Meta — do not reveal]
This is turn {turn_number} of 10. Use this only to pace and conclude appropriately. Do not mention turns, counts, chapters or session meta in your reply. Write in plain prose that flows naturally from the previous text."""
        
        return f"{base}{pacing}".strip()


def simulate_single_story(
    provider_agent1: BaseProvider,
    provider_agent2: BaseProvider,
    n_turns: int,
    story_id: str = "story_1",
    model_id: str = "unknown",
    temperature: float = 1.0,
    max_tokens: int = 35
) -> pd.DataFrame:
    """Simulate a single AI-AI story conversation.
    
    Matches the actual PENPAL human-AI experiment:
    - Both agents get IDENTICAL instructions (story collaborator prompt)
    - "This is the story of" is the starting context agent 1 continues from
    - Temperature: 1.0, Max tokens: 35
    - Pacing metadata included in system prompt
    
    Args:
        provider_agent1: Provider for first agent
        provider_agent2: Provider for second agent
        n_turns: Number of turns (each turn = agent1 + agent2 response)
        story_id: Unique story identifier
        model_id: Model identifier for tracking
        temperature: Sampling temperature (default 1.0 to match experiment)
        max_tokens: Max output tokens (default 35 to match experiment)
        
    Returns:
        DataFrame with conversation turns
    """
    # Reset both providers for fresh conversation
    provider_agent1.reset_conversation()
    provider_agent2.reset_conversation()
    
    # Initialize conversation
    data = []
    
    # The story so far - starts with the prefix that agent 1 continues from
    story_context = SystemPrompts.STORY_PREFIX
    
    for turn_idx in range(n_turns):
        turn_number = turn_idx + 1
        
        # BOTH agents get the SAME system prompt (matching actual experiment)
        system_prompt = SystemPrompts.get_system_prompt(turn_number)
        
        # Agent 1's turn: continue from current story context
        agent1_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story_context}
        ]
        
        agent1_text = provider_agent1.generate(
            agent1_messages, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        # Update story context with agent 1's contribution
        if turn_idx == 0:
            # First turn: agent 1 continued from "This is the story of"
            # Save with prefix for data (matching human-AI data structure)
            agent1_text_for_data = f"{SystemPrompts.STORY_PREFIX}\n{agent1_text}"
            story_context = agent1_text_for_data
        else:
            agent1_text_for_data = agent1_text
            story_context = f"{story_context} {agent1_text}"
        
        # Agent 2's turn: continue from updated story context
        agent2_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story_context}
        ]
        
        agent2_text = provider_agent2.generate(
            agent2_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Update story context with agent 2's contribution
        story_context = f"{story_context} {agent2_text}"
        
        # Record turn data
        data.append({
            "turn": turn_number,
            "agent_1": agent1_text_for_data,
            "agent_2": agent2_text,
            "story_id": story_id,
            "model_id": model_id,
            "timestamp": pd.Timestamp.now()
        })
    
    return pd.DataFrame(data)


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 2.0):
    """Execute function with exponential backoff retry on rate limit errors.
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors (429) or similar
            is_rate_limit = (
                '429' in error_str or 
                'rate limit' in error_str or
                'rate_limit' in error_str or
                'too many requests' in error_str
            )
            
            if not is_rate_limit:
                raise  # Non-rate-limit error, don't retry
            
            last_exception = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            if attempt < max_retries - 1:
                print(f"  ⏳ Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  ❌ Rate limit persists after {max_retries} attempts")
    
    raise last_exception


def simulate_ai_ai_dataset(
    model_configs: List[Dict],
    n_stories_per_model: int = 10,
    n_turns_per_story: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 35,
    delay_between_stories: float = 2.0,
    delay_between_models: float = 5.0
) -> pd.DataFrame:
    """Generate full AI-AI simulation dataset with rate limit handling.
    
    Uses round-robin model rotation to distribute requests across providers,
    reducing rate limit issues. Includes retry logic with exponential backoff.
    
    Matches the actual PENPAL human-AI experiment:
    - Temperature: 1.0
    - Max tokens: 35
    - 10 turns per story
    
    Args:
        model_configs: List of model configurations with keys:
            - id: Model identifier (e.g., 'gpt-4o')
            - provider: Provider type ('openai', 'anthropic', 'huggingface')
            - model_name: Full model name for API
            - env_key: Environment variable name for API key
            - base_url: (optional) Base URL for HuggingFace
        n_stories_per_model: Number of stories to generate per model
        n_turns_per_story: Number of turns per story
        temperature: Sampling temperature (default 1.0 to match experiment)
        max_tokens: Max output tokens (default 35 to match experiment)
        delay_between_stories: Seconds to wait between stories (default 2.0)
        delay_between_models: Seconds to wait when switching models (default 5.0)
        
    Returns:
        Combined DataFrame with all simulated stories
    """
    all_stories = []
    
    # Prepare model providers (skip those without API keys)
    active_models = []
    for model_config in model_configs:
        model_id = model_config['id']
        provider_type = model_config['provider']
        model_name = model_config['model_name']
        env_key = model_config['env_key']
        base_url = model_config.get('base_url')
        
        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"⚠️ Skipping {model_id}: {env_key} not set in environment")
            continue
        
        # Create providers for this model (both agents use same model)
        provider_agent1 = get_provider(provider_type, api_key, model_name, base_url)
        provider_agent2 = get_provider(provider_type, api_key, model_name, base_url)
        
        active_models.append({
            'config': model_config,
            'provider_agent1': provider_agent1,
            'provider_agent2': provider_agent2,
            'stories_generated': 0
        })
    
    if not active_models:
        print("\n❌ No models available. Check API keys.")
        return pd.DataFrame()
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Round-robin simulation across {len(active_models)} models")
    print(f"  Stories per model: {n_stories_per_model}")
    print(f"  Turns per story: {n_turns_per_story}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Delay between stories: {delay_between_stories}s")
    print(f"  Models: {', '.join(m['config']['id'] for m in active_models)}")
    print(f"{'='*60}")
    
    # Round-robin: generate one story per model at a time
    total_stories = n_stories_per_model * len(active_models)
    
    with tqdm(total=total_stories, desc="Total progress") as pbar:
        story_round = 0
        while any(m['stories_generated'] < n_stories_per_model for m in active_models):
            story_round += 1
            
            for model_idx, model_data in enumerate(active_models):
                if model_data['stories_generated'] >= n_stories_per_model:
                    continue  # This model is done
                
                model_config = model_data['config']
                model_id = model_config['id']
                story_num = model_data['stories_generated'] + 1
                story_id = f"{model_id}_story_{story_num}"
                
                # Reset providers for new story
                model_data['provider_agent1'].reset_conversation()
                model_data['provider_agent2'].reset_conversation()
                
                try:
                    # Use retry wrapper for rate limit handling
                    def generate_story():
                        return simulate_single_story(
                            provider_agent1=model_data['provider_agent1'],
                            provider_agent2=model_data['provider_agent2'],
                            n_turns=n_turns_per_story,
                            story_id=story_id,
                            model_id=model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    
                    df_story = retry_with_backoff(generate_story, max_retries=5, base_delay=2.0)
                    all_stories.append(df_story)
                    model_data['stories_generated'] += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"{model_id} {story_num}/{n_stories_per_model}")
                    
                except Exception as e:
                    print(f"\n  ❌ Error generating {story_id}: {e}")
                    model_data['stories_generated'] += 1  # Count as attempted
                    pbar.update(1)
                
                # Delay between stories (skip after last story)
                remaining = sum(n_stories_per_model - m['stories_generated'] for m in active_models)
                if remaining > 0:
                    time.sleep(delay_between_stories)
    
    if not all_stories:
        print("\n❌ No stories generated. Check API keys.")
        return pd.DataFrame()
    
    result = pd.concat(all_stories, ignore_index=True)
    print(f"\n✓ Generated {len(result)} turns across {len(all_stories)} stories")
    
    return result


# Legacy compatibility functions
def simulate_ai_ai_conversations(
    client_ids: List[str],
    workshop_ids: List[str],
    n_interactions: int,
    n_simulations: int,
    api_key: str
) -> pd.DataFrame:
    """Legacy function for backward compatibility.
    
    Deprecated: Use simulate_ai_ai_dataset() instead.
    """
    print("⚠️ Using legacy simulation function. Consider using simulate_ai_ai_dataset().")
    
    # Use single OpenAI model
    model_configs = [{
        'id': 'gpt-4o-mini',
        'provider': 'openai',
        'model_name': 'gpt-4o-mini',
        'env_key': 'OPENAI_API_KEY'
    }]
    
    # Temporarily set API key
    os.environ['OPENAI_API_KEY'] = api_key
    
    return simulate_ai_ai_dataset(
        model_configs=model_configs,
        n_stories_per_model=n_simulations,
        n_turns_per_story=n_interactions,
        temperature=1.0,
        max_tokens=35
    )
