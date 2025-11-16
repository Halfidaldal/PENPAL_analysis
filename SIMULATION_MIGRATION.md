# Simulation Module Migration Complete

## What Was Migrated

Successfully migrated `Berlin/src/simulate_baseline_data.py` into the modular structure.

### Files Created/Modified

1. **`src/nes/simulation.py`** - Now contains full implementation:
   - `GenRequest` class - OpenAI conversation generator
   - `SystemPrompts` class - German workshop prompts (Utopian/Dystopian)
   - `simulate_ai_ai_conversations()` - High-level function for batch simulation
   - `add_humanlike_variation()` - Placeholder for future variation

2. **`scripts/05_simulate_baseline.py`** - New pipeline script:
   - Command-line interface for simulation
   - Loads config from `config.yaml`
   - Accepts API key via flag or environment variable
   - Saves output to `data/processed/simulated_ai_ai_baseline.csv`

3. **`config.yaml`** - Updated simulation section:
   - `n_turns_per_story: 10`
   - `model_name: "gpt-4o"`
   - `client_ids: ["1", "2"]`
   - `workshop_ids: ["1", "2", "3"]`

4. **`environment/requirements_clean.txt`** - Added dependency:
   - `openai>=1.0.0`

## Key Changes from Original

### What Stayed the Same
✅ Exact same conversation logic (`GenRequest` class)
✅ Same system prompts (German workshop prompts)
✅ Same Utopian/Dystopian logic based on client_id
✅ Same OpenAI API integration (Conversations API)
✅ Same output format (DataFrame with turn/user/ai/etc.)

### What Changed (Improvements)
✨ **Modular structure**: Logic in `src/nes/simulation.py`, execution in `scripts/05_simulate_baseline.py`
✨ **Configuration**: Parameters from `config.yaml` instead of hardcoded
✨ **Reusable**: Can import functions into notebooks or other scripts
✨ **Type hints**: Added proper type annotations
✨ **Documentation**: Comprehensive docstrings
✨ **Flexible API key**: Can pass via flag, env var, or config
✨ **Consistent I/O**: Uses `nes.io.save_csv()` for standardized output

## How to Use

### Run Simulation from Command Line

```bash
# Set API key as environment variable (recommended)
export OPENAI_API_KEY=your_key_here
python scripts/05_simulate_baseline.py

# Or pass API key directly
python scripts/05_simulate_baseline.py --api-key your_key_here

# Customize parameters
python scripts/05_simulate_baseline.py \
    --n-interactions 5 \
    --client-ids 1 2 \
    --workshop-ids 1 2 3
```

### Use in Python/Notebook

```python
from nes.simulation import simulate_ai_ai_conversations, SystemPrompts
import os

# Get API key
api_key = os.environ['OPENAI_API_KEY']

# Simulate one conversation
df = simulate_ai_ai_conversations(
    client_ids=['1'],
    workshop_ids=['2'],
    n_interactions=10,
    api_key=api_key
)

# Or use lower-level classes directly
from nes.simulation import GenRequest

prompt1 = SystemPrompts.get_system_prompt_ai1('2')
prompt2 = SystemPrompts.get_system_prompt_ai2('1', '2')

df = GenRequest.simulate_data(
    n_interactions=10,
    systemprompt1=prompt1,
    systemprompt2=prompt2,
    client_id='1',
    workshop_id='2',
    api_key=api_key
)
```

## Output Format

The simulation produces a DataFrame with columns:
- `turn` - Turn number (1-10)
- `user` - Text from AI 1 (user simulator)
- `ai` - Text from AI 2 (story assistant)
- `client_id` - Client identifier
- `workshop_id` - Workshop identifier
- `conversation_id` - OpenAI conversation ID
- `timestamp` - When the turn was generated

Saved to: `data/processed/simulated_ai_ai_baseline.csv`

## Migration Checklist

- [x] Extract `GenRequest` class to module
- [x] Extract `SystemPrompts` class to module
- [x] Create high-level `simulate_ai_ai_conversations()` function
- [x] Create `scripts/05_simulate_baseline.py` entry point
- [x] Add OpenAI dependency to requirements
- [x] Update config.yaml with simulation parameters
- [x] Add type hints and docstrings
- [x] Test that API key handling works
- [x] Preserve exact same functionality

## Next Steps (Optional)

If you want to further enhance the simulation module:

1. **Add variation function**: Implement `add_humanlike_variation()` to add typos, etc.
2. **Add retry logic**: Handle API rate limits and transient errors
3. **Add progress saving**: Save intermediate results in case of failure
4. **Add cost estimation**: Warn user about expected API costs before running
5. **Add batch support**: Run multiple simulations in parallel

## Testing

To verify the migration works correctly:

```bash
# Test with minimal simulation (saves API costs)
export OPENAI_API_KEY=your_key
python scripts/05_simulate_baseline.py --n-interactions 2 --client-ids 1 --workshop-ids 1

# Check output
ls -lh data/processed/simulated_ai_ai_baseline.csv
head data/processed/simulated_ai_ai_baseline.csv
```

Expected: 2 rows (1 conversation × 2 turns) with German text in user/ai columns.

## Important Notes

⚠️ **API Costs**: Each simulation uses OpenAI API and incurs costs
- ~6 conversations (2 clients × 3 workshops) with 10 turns each = 60 API calls
- Estimated cost: ~$0.50-2.00 depending on model and token usage

⚠️ **API Key Security**: 
- Never commit API keys to git
- Use environment variables or secret management
- The old script had hardcoded key - removed in migration

⚠️ **Rate Limits**:
- OpenAI has rate limits (requests per minute)
- For large simulations, consider adding delays between calls
- Or use batch API if available

---

**Status**: ✅ Migration complete! The simulation functionality is now a proper module with a clean script interface.
