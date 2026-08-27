# pplyz - Development Notes

## Project Overview

pplyz (short for "LLM Analyser") is a Python-based command-line tool designed to process CSV data using Large Language Models (LLMs). The tool enables users to augment their datasets with AI-generated structured outputs by processing each row sequentially through an LLM API.

**Version 0.2.0** introduces LiteLLM integration for multi-provider support, enabling seamless switching between Gemini, OpenAI, Anthropic, and 100+ other LLM providers with enforced JSON structured output.

## Architecture

### Core Components

The project is structured into modular components for maintainability and separation of concerns:

```
pplyz/
├── config.py       # Configuration constants and settings
├── llm_client.py   # LLM API client with retry logic
├── processor.py    # CSV processing orchestration
└── schemas.py      # Output schema definitions
```

### Design Decisions

#### 1. Sequential Processing

**Decision**: Process rows one at a time instead of parallel processing.

**Rationale**:
- API rate limits are the primary bottleneck, not CPU
- Simplifies error handling and recovery
- Reduces risk of rate limit violations
- Makes progress tracking more straightforward
- Lower memory footprint for large datasets

**Trade-off**: Slower processing, but more reliable and cost-effective.

#### 2. Interactive Prompt Input

**Decision**: Collect analysis prompts interactively rather than via command-line arguments.

**Rationale**:
- Prompts can be long and complex
- Shell escaping for multi-line prompts is error-prone
- Better user experience for iterative prompt refinement
- Cleaner command-line interface

**Alternative Considered**: Config file-based prompts (rejected for initial version to keep it simple).

#### 3. LiteLLM for Multi-Provider Support

**Decision**: Use LiteLLM instead of provider-specific SDKs.

**Rationale**:
- **Unified Interface**: Single API for 100+ LLM providers
- **JSON Mode**: Built-in `response_format={"type": "json_object"}` support
- **Provider Flexibility**: Easy to switch between providers without code changes
- **Future-Proof**: New providers are supported automatically
- **Cost Optimization**: Can easily switch to cheaper models for testing

**Trade-off**: Adds another dependency, but the benefits far outweigh the cost.

#### 4. Simple Dictionary-Based Output Schema

**Decision**: Use simple JSON dictionaries for structured output with enforced JSON mode.

**Rationale**:
- Minimal boilerplate code
- Flexible for different use cases
- LLM determines output structure based on prompt
- JSON mode ensures reliable parsing
- Easy to extend and modify
- Lower learning curve for users

**Trade-off**: Less type safety than Pydantic, but more flexibility.

#### 5. Automatic JSON Parsing with Cleanup

**Decision**: Use JSON mode + automatic cleanup of markdown code blocks.

**Rationale**:
- JSON mode (`response_format={"type": "json_object"}`) enforces JSON output
- Fallback cleanup handles edge cases where LLMs still add markdown
- Improves reliability without requiring perfect prompt engineering
- Transparent to users
- Handles common formatting variations

**Note**: JSON mode is natively supported by OpenAI and Gemini. LiteLLM handles other providers gracefully.

## API Client Design

### LiteLLM Integration

The `LLMClient` uses LiteLLM's `completion()` function for unified multi-provider access:

```python
response = completion(
    model=self.model_name,
    messages=[...],
    response_format={"type": "json_object"}  # Enforced JSON mode
)
```

**Key Features**:
- **Provider Detection**: Automatically detects provider from model name
- **API Key Management**: Uses standard environment variables per provider
- **JSON Mode**: Enforces structured output when supported
- **Graceful Degradation**: Falls back to prompt-based JSON for unsupported providers

### Retry Strategy

The `LLMClient` uses the `tenacity` library for sophisticated retry logic:

```python
@retry(
    retry=retry_if_exception_type((
        RateLimitError,
        ServiceUnavailableError,
        APIError,
        Timeout,
    )),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=1, max=60),
    reraise=True,
)
```

**Key Features**:
- Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
- Retries LiteLLM exception types (unified across providers)
- Maximum 5 retry attempts
- Re-raises exception if all retries exhausted

### Rate Limiting

Built-in rate limiting prevents API quota violations:

```python
def _rate_limit_delay(self):
    elapsed = time.time() - self.last_request_time
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
```

- 0.5 second minimum delay between requests
- Configurable via `config.py`
- Applied before each API call

## CSV Processing

### Error Handling Strategy

The processor continues processing even when individual rows fail:

1. **Catch Exceptions**: Errors for individual rows are caught and logged
2. **Empty Placeholders**: Failed rows get empty dictionaries to maintain alignment
3. **Error Summary**: All errors are reported at the end
4. **Partial Success**: Successfully processed rows are saved even if some fail

**Benefits**:
- Long-running jobs don't fail completely due to single row errors
- Users can identify and manually fix problematic rows
- Better UX for large datasets

### Preview Mode

Preview mode allows users to test prompts before full processing:

```bash
pplyz data.csv --input col1,col2 --preview
```

**Features**:
- Processes only first N rows (default: 3)
- Shows input and output for each row
- Doesn't save to file
- Helps refine prompts before expensive full runs

## Configuration Management

### Environment Variables

API keys are managed via environment variables or TOML config files:

- **Security**: Keeps secrets out of code
- **Flexibility**: Easy to switch between dev/prod environments
- **Standard Practice**: Follows twelve-factor app methodology

### Configuration File

`config.py` centralizes all configuration:

```python
DEFAULT_MODEL = "gemini-2.5-flash-lite"
RETRY_BACKOFF_SCHEDULE = [1, 2, 3, 5, 10, 10, 10, 10, 10]
MAX_RETRIES = len(RETRY_BACKOFF_SCHEDULE) + 1
REQUEST_DELAY = 0.5
```

**Benefits**:
- Single source of truth
- Easy to modify behavior without code changes
- Type-safe constants
- Well-documented settings

## Testing Considerations

### Manual Testing Strategy

For initial development, focus on manual testing:

1. **Preview Mode**: Test prompts on sample data
2. **Small Datasets**: Use small CSVs for integration testing
3. **Error Injection**: Test with invalid API keys, malformed prompts
4. **Rate Limit Testing**: Process enough rows to trigger rate limits

### Future Testing Improvements

For production use, consider:

- **Unit Tests**: Test individual components (client, processor)
- **Mock API**: Use mock LLM responses for deterministic testing
- **Integration Tests**: End-to-end tests with sample data
- **Performance Tests**: Benchmark processing speed and memory usage

## Potential Enhancements

### Short-term Improvements

1. **Resume Capability**: Save progress and resume interrupted jobs
   - Checkpoint processed rows
   - Skip already-processed rows on restart

2. **Batch Preview**: Show multiple rows side-by-side for comparison
   - Better for prompt tuning
   - Identify edge cases

3. **Custom Output Schemas**: Support Pydantic models for type safety
   - Optional feature for advanced users
   - Better validation of LLM outputs

4. **Progress Bar**: Visual progress indicator using `tqdm`
   - Estimated time remaining
   - Success/failure counts

### Long-term Enhancements

1. **Multiple LLM Support**: Add OpenAI, Anthropic, local models
   - Unified interface
   - Model comparison mode

2. **Parallel Processing**: Optional parallel mode with rate limiting
   - Semaphore-based concurrency control
   - Configurable worker count

3. **Cost Tracking**: Estimate and track API costs
   - Token counting
   - Cost per row
   - Total job cost

4. **Streaming Mode**: Process large files without loading entirely into memory
   - Chunk-based processing
   - Lower memory footprint

## Common Pitfalls and Solutions

### 1. Rate Limiting

**Problem**: API returns 429 errors despite retry logic.

**Solution**:
- Increase `REQUEST_DELAY` in config
- Reduce batch size (already at 1)
- Use a higher-tier API plan

### 2. Invalid JSON Responses

**Problem**: LLM returns text that isn't valid JSON.

**Solution**:
- Make prompt more specific about JSON requirements
- Add examples of expected output format to prompt
- Use temperature=0 for more deterministic outputs (future feature)

### 3. Long-Running Jobs

**Problem**: Processing large datasets takes too long.

**Solution**:
- Use preview mode to validate approach first
- Process in chunks (split CSV manually)
- Consider parallel processing (future feature)

### 4. Memory Issues

**Problem**: Large CSVs consume too much memory.

**Solution**:
- Use chunk-based processing (future feature)
- Process in smaller batches
- Increase available memory

## Development Workflow

### Adding a New Feature

1. **Update Config**: Add any new constants to `config.py`
2. **Implement Logic**: Add feature in appropriate module
3. **Update CLI**: Modify `main.py` to expose new options
4. **Document**: Update `README.md` with usage examples
5. **Test**: Write and run tests (see Testing Workflow below)

### Testing Workflow

When making code changes, always follow this workflow:

1. **Write Tests First**: Add or update tests in `tests/` directory before implementing changes
   - New features: Write tests covering happy path and edge cases
   - Bug fixes: Add regression tests to prevent recurrence
   - Refactoring: Maintain existing test coverage

2. **Run Relevant Tests**: Run affected test modules during development
   ```bash
   uv run pytest tests/test_module.py -v
   ```

3. **Run Full Test Suite**: Before committing, run all tests to catch regressions
   ```bash
   uv run pytest tests/ -v
   ```

4. **Fix Failing Tests**: Ensure all tests pass before committing changes
   - Update test code if API signatures change
   - Fix implementation if tests reveal bugs
   - Never commit with failing tests

5. **Update Test Documentation**: Keep test docstrings clear and descriptive
   - Explain what is being tested
   - Document expected behavior
   - Note any special test setup requirements

### Automated Testing with Pre-commit Hooks

This project uses pre-commit hooks to automatically run tests before each commit.

**Setup (one-time)**:
```bash
# Install pre-commit hooks
uv run pre-commit install
```

**Usage**:
- Tests run automatically when you commit
- If tests fail, the commit is blocked
- Fix the issues and try committing again

**Bypass hooks (use sparingly)**:
```bash
# Skip pre-commit hooks for a specific commit
git commit --no-verify -m "message"
```

**Manual pre-commit run**:
```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run pytest --all-files
```

**What gets checked**:
- ✅ All pytest tests must pass
- ✅ Code formatting (Ruff)
- ✅ Code linting (Ruff)
- ✅ Trailing whitespace removal
- ✅ YAML file validation
- ✅ No large files added
- ✅ No merge conflicts
- ✅ No private keys committed

**Test Coverage Goals**:
- Aim for high coverage of critical paths
- All public APIs should have tests
- Edge cases and error conditions must be tested
- Integration points between modules need coverage

**Running Specific Test Categories**:
```bash
# Run only unit tests
uv run pytest -m unit

# Run with coverage report
uv run pytest --cov=llm_analyser tests/

# Run with verbose output
uv run pytest -v -s
```

### Code Style

- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Error Messages**: Clear, actionable error messages
- **Logging**: Use print statements for user feedback (simple approach)

### Dependencies

Minimal dependency philosophy:
- `litellm`: Multi-provider LLM API client (replaces provider-specific SDKs)
- `pandas`: CSV processing (industry standard)
- `tenacity`: Retry logic (battle-tested)
- `python-dotenv`: Environment management (standard practice)
- `pydantic`: Schema validation for structured output

**Rationale**: Each dependency must provide significant value and be well-maintained. LiteLLM consolidates what would otherwise be multiple provider-specific dependencies.

## Deployment Considerations

### Distribution

Current approach: Git repository + uv

**Future Options**:
- PyPI package: `pip install pplyz`
- Docker container: Isolated environment
- Pre-built binaries: PyInstaller for non-Python users

### Configuration for Production

1. **API Key Management**: Use secrets manager (AWS Secrets Manager, etc.)
2. **Logging**: Replace print statements with proper logging framework
3. **Monitoring**: Add metrics and alerting
4. **Error Tracking**: Integrate with error tracking service (Sentry, etc.)

## Performance Characteristics

### Time Complexity

- **Per Row**: O(1) API call + O(m) where m is response size
- **Total**: O(n) where n is number of rows
- **Bottleneck**: API latency (typically 1-3 seconds per request)

### Space Complexity

- **Memory**: O(n) - entire DataFrame held in memory
- **Storage**: O(n * m) where m is average output size

### Typical Performance

With default settings:
- ~100-120 rows per minute
- Limited by REQUEST_DELAY (0.5s) and API latency
- ~6000-7000 rows per hour

## Security Considerations

1. **API Key Storage**: Never commit secret config files (`pplyz.local.toml`, `.env`, etc.)
2. **Input Validation**: Validate column names and file paths
3. **Output Sanitization**: Be cautious with user-generated prompts
4. **Rate Limiting**: Prevents accidental cost overruns

## Maintenance Notes

### Regular Tasks

- **Dependency Updates**: Check for security updates monthly
- **API Changes**: Monitor Gemini API changelog for breaking changes
- **Cost Monitoring**: Review API usage and costs

### Known Issues

None currently. Track issues in GitHub Issues when available.

## Schema Design Patterns

### Understanding Structured Output Validation

LLM Analyser supports three levels of output validation through LiteLLM:

1. **JSON Syntax Validation** (basic mode)
   - Ensures valid JSON structure
   - No field or type validation
   - Works with all providers

2. **Type Validation** (with `--output`)
   - Enforces field presence and basic types
   - Boolean, int, float strictly enforced
   - String fields accept any text (no enum constraint)

3. **Enum/Pattern Validation** (limited support)
   - Only supported by select models (OpenAI GPT-4o)
   - Not universally available across providers
   - **Requires post-processing for most models**

### Type Enforcement Reality

Based on real-world testing with multiple providers:

| Field Type | Enforcement Level | Validation Behavior |
|------------|------------------|---------------------|
| `bool` | ✅ **Strict** | Only `true`/`false` allowed |
| `int` | ✅ **Strict** | Only integers allowed |
| `float` | ✅ **Strict** | Only numbers allowed |
| `str` | ⚠️ **Flexible** | Any text accepted (no enum) |
| `list[str]` | ⚠️ **Flexible** | JSON array of strings |

**Key Insight**: String fields with `--output 'relation:str'` only enforce that the value is a string. They do NOT enforce enum-like constraints (e.g., "must be high/middle/low").

### Recommended Schema Patterns

#### Pattern 1: Boolean Gates (Most Reliable)

For binary classification tasks, use boolean fields:

```bash
--output "is_relevant:bool,confidence:str,reason:str"
```

**Advantages**:
- Guaranteed `true`/`false` values across ALL providers
- No ambiguous responses like "yes", "maybe", "N/A"
- Perfect for filtering and downstream processing
- Works identically on Gemini, OpenAI, Claude, etc.

**Example prompt**:
```
Determine if this article is relevant to DISH research (true/false).
Provide confidence level (high/medium/low) and reason.
```

#### Pattern 2: String Fields (Requires Care)

For multi-level classification (high/middle/low):

```bash
--output "relation:str,reason:str"
```

**Reality**:
- LLMs may return unexpected values
- No provider enforces enum constraints for string fields
- Even explicit prompts are often ignored

**Best Practice**:
- Use very explicit prompts listing all allowed values
- Consider post-processing for production use
- Use boolean flags when possible (see Pattern 3)

#### Pattern 3: Hybrid Approach (Production Ready)

Combine boolean gates with descriptive string fields:

```bash
--output "is_dish_related:bool,relevance_level:str,explanation:str"
```

**Workflow**:
1. Filter by boolean field: `df[df['is_dish_related'] == True]`
2. Review relevance_level for additional context
3. Use explanation for manual review of edge cases

**Benefits**:
- Reliable filtering via boolean
- Rich context via string fields
- Consistent behavior across providers

### Provider-Specific Behavior

| Provider | Boolean Fields | String Enums | Format Compliance |
|----------|---------------|--------------|-------------------|
| **Gemini Flash** | ✅ Excellent | ❌ Poor | May return unexpected string values |
| **OpenAI GPT-4o** | ✅ Excellent | ✅ Good | Best enum support (via Pydantic) |
| **Anthropic Claude** | ✅ Excellent | ⚠️ Moderate | Good with strong prompts |
| **Ollama (local)** | ⚠️ Model-dependent | ❌ Poor | JSON mode only; small models drift. Test with `--preview` |

**Recommendation**: For multi-provider compatibility, design schemas assuming string fields will not enforce enums.

### Schema Design Checklist

When designing your schema:

1. ✅ **Use boolean for binary decisions** - Most reliable across all providers
2. ⚠️ **String fields may return unexpected values** - Plan accordingly
3. ✅ **Combine types strategically** - Boolean gates + string descriptions
4. ⚠️ **Test prompts in preview mode** - Check actual output before full run
5. ✅ **Use explicit prompts** - List all allowed values clearly
6. ⚠️ **Don't rely on prompt compliance alone** - Even "MUST" instructions get ignored

## Lessons Learned

### Real-World Findings from Production Use

#### 1. String Field Enum Constraints Are Not Enforced

**Discovery**: String fields with expected enum-like values often return unexpected outputs.
- Expected: `"relation": "high"` | `"middle"` | `"low"`
- Actual: Various unexpected values like "unrelated", "no", "N/A", etc.

**Lesson**: `--output 'field:str'` only enforces type (string), not enum values.

#### 2. Boolean Fields Provide Strict Enforcement

**Discovery**: During investigation of why string enums failed, tested boolean fields:
- Boolean fields return ONLY `true` or `false`
- No ambiguous values possible
- Consistent across all tested providers (Gemini, OpenAI, Claude)

**Lesson**: For any yes/no decision, always use `"field": "bool"` instead of `"field": "str"` with prompt-based instructions.

**Impact**: This became a primary recommendation in README.md Best Practices section.

#### 3. Prompt Compliance Varies Dramatically by Model

**Discovery**: Same exact prompt with different models can produce varying output formats.
- Some models may ignore CRITICAL, MUST, and example output format
- Newer models don't necessarily follow instructions better for structured output

**Lesson**: Never rely solely on prompt engineering. Test thoroughly with preview mode.

#### 4. Multi-Provider Support Requires Lowest Common Denominator

**Discovery**: LiteLLM's `supports_response_schema()` returns different results:
- OpenAI GPT-4o: Supports Pydantic models with enums
- Gemini models: Only support basic JSON mode
- Other providers: Varies widely

**Lesson**: For tools targeting multiple providers, design around the most limited provider's capabilities.

**Trade-off**: Lose strict validation for gain in provider flexibility.

#### 5. Resume Functionality Needs Robust Empty Detection

**Discovery**: During code review, identified edge cases in resume logic:
- Empty strings `""` vs. `NaN` vs. `"{}"` (empty JSON)
- Rows could be skipped incorrectly or processed unnecessarily

**Lesson**: Robust empty detection requires checking multiple conditions:

```python
def _should_process_row(self, row, new_column_names):
    for col in new_column_names:
        if col in row.index:
            value = row[col]
            if pd.isna(value) or value == "":
                return True
            try:
                parsed = json.loads(str(value))
                if not parsed or parsed == {}:
                    return True
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return not any(col in row.index for col in new_column_names)
```

**Impact**: Prevents data loss and unnecessary API calls in resume mode.

#### 6. Logging Infrastructure Matters for Debugging

**Discovery**: Initial implementation used `print()` statements throughout.
- Hard to filter user-facing messages vs. debug info
- No log levels for different severity
- Difficult to redirect to files for long-running jobs

**Lesson**: Even for "simple" CLI tools, use proper logging from the start:

```python
import logging
logger = logging.getLogger(__name__)

# User-facing
logger.info("Processing row 10/100...")

# Debugging
logger.warning("Unknown provider detected...")
```

**Benefit**: Easy to add file logging, log levels, and structured logging later.

#### 7. Model Comparison Requires Controlled Experiments

**Discovery**: Different model versions may produce different output formats.

**Lesson**: For meaningful model comparison:
1. Use identical prompts
2. Same dataset and random seed (if applicable)
3. Compare semantic meaning, not just format compliance
4. Use preview mode to validate outputs

**Insight**: Newer model versions may be optimized for different objectives than format compliance.

### Technical Debt and Future Improvements

Based on lessons learned:

1. **Add schema validation warnings**:
   - Warn users when using string fields for enum-like values
   - Suggest boolean alternatives

2. **Improve prompt templates**:
   - Provide model-specific prompt templates
   - Include examples that work better for each provider

3. **Add output validation**:
   - Option to validate outputs against expected patterns
   - Flag unexpected values during processing

4. **Better progress tracking**:
   - Show validation warnings in real-time
   - Alert user if outputs deviate from expectations

### Documentation Philosophy

**Key Realization**: Users expect structured output to be more strict than it actually is.

**Solution**: README.md now explicitly documents:
1. Three levels of validation (syntax, type, enum)
2. Type enforcement table showing which types are strict
3. Provider-specific behavior comparison
4. Multiple schema design patterns with trade-offs

**Goal**: Set realistic expectations and provide actionable workarounds.

## Version History

### v0.1.6 (Current)
- **Ollama / local model support**: `--model ollama_chat/<name>` runs against a local
  Ollama server with no API key required
- **`--api-base` flag** (+ `PPLYZ_API_BASE` / `OLLAMA_API_BASE` env, `default_api_base`
  in `config.toml`): point pplyz at Ollama or any OpenAI-compatible local server
- **JSON mode for schema-less providers**: providers that expose `response_format` but
  not full schema support (e.g. Ollama) now get `{"type": "json_object"}` enforcement
- **Actionable connection errors**: `APIConnectionError` is turned into a hint about
  `ollama serve` / `ollama pull` / `--api-base`
- No new dependencies (LiteLLM already ships Ollama support)

### v0.2.0
- **LiteLLM Integration**: Replaced `google-generativeai` with LiteLLM
- **Multi-Provider Support**: Supports Gemini, OpenAI, Anthropic, and 100+ providers
- **Enforced JSON Mode**: Uses `response_format={"type": "json_object"}`
- **Provider Auto-Detection**: Automatically detects provider from model name
- **Model Listing**: Added `--list` (`-l`) command

### v0.1.0
- Initial release with Gemini-only support
- Basic CSV processing with LLM-generated columns
- Retry logic and rate limiting
- Preview mode

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [uv Documentation](https://github.com/astral-sh/uv)
