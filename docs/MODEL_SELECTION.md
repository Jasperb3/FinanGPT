# Model Selection Guide for FinanGPT

**Last Updated**: 2025-11-10
**Version**: 1.0

## Overview

FinanGPT uses local LLMs via Ollama to convert natural language queries into SQL. Not all models perform equally well for this task. This guide helps you choose the right model.

## Quick Recommendation

**Best Overall**: `qwen2.5-coder:14b` or `qwen3:14b`

Update your `.env` file:
```bash
MODEL_NAME=qwen2.5-coder:14b
```

Then pull the model:
```bash
ollama pull qwen2.5-coder:14b
```

## Model Comparison

### Tier 1: Excellent (Recommended)

| Model | Size | Strengths | Best For |
|-------|------|-----------|----------|
| **qwen2.5-coder:14b** | 8.8GB | Code generation, instruction following | SQL generation, complex queries |
| **qwen3:14b** | 8.8GB | General purpose, good reasoning | Balanced performance |
| **deepseek-coder-v2:16b** | 9.4GB | Strong SQL understanding | Complex joins, window functions |

### Tier 2: Good

| Model | Size | Strengths | Limitations |
|-------|------|-----------|-------------|
| **mistral-small3.2:latest** | 8.9GB | Instruction following | Slower inference |
| **codellama:13b** | 7.4GB | Code-focused | Less natural language understanding |

### Tier 3: Acceptable (with caveats)

| Model | Size | Issues | Workarounds |
|-------|------|--------|-------------|
| **gpt-oss:latest** | Various | Generates prose instead of SQL, hallucinates table names | Use templates, avoid complex queries |
| **phi4:latest** | 8.7GB | Inconsistent SQL formatting | Works better with simple queries |

## Common Issues by Model

### `gpt-oss:latest` (Current Default)

**Issues**:
- ❌ Returns markdown tables instead of SQL
- ❌ Hallucinates non-existent tables (`stocks` instead of `company.metadata`)
- ❌ Uses wrong column names (`name` instead of `longName`)
- ❌ Ignores "SQL-only" instructions

**Why**: This model is optimized for conversational helpfulness, not strict code generation.

**Solution**: Switch to a code-specialized model

### `qwen3:14b` (Recommended)

**Issues**:
- ✅ Follows SQL-only instructions
- ✅ Uses correct table and column names
- ✅ Respects schema constraints
- ⚠️  May need 16GB+ RAM for best performance

**Example**:
```bash
# Update .env
echo "MODEL_NAME=qwen3:14b" >> .env

# Pull model
ollama pull qwen3:14b

# Test
python finangpt.py query "Show me tech stocks with P/E ratio < 20"
```

## Model Requirements

### System Requirements

| Model Size | Minimum RAM | Recommended RAM | GPU VRAM |
|------------|-------------|-----------------|----------|
| 7B | 8 GB | 16 GB | 6 GB (optional) |
| 13-14B | 16 GB | 32 GB | 8 GB (optional) |
| 16B+ | 24 GB | 48 GB | 12 GB (optional) |

### Inference Speed

Approximate query response times (on CPU):

| Model | Apple Silicon (M1/M2) | Intel i7 (32GB RAM) | AMD Ryzen 9 |
|-------|----------------------|---------------------|-------------|
| qwen2.5-coder:14b | 3-5s | 5-8s | 4-7s |
| qwen3:14b | 3-5s | 5-8s | 4-7s |
| mistral-small3.2 | 4-6s | 6-10s | 5-9s |

## Testing Your Model

Run these test queries to evaluate model performance:

### Test 1: Simple Valuation Filter
```bash
python finangpt.py query "Show me tech stocks with P/E ratio < 20"
```

**Expected**: Should use `company.metadata` and `valuation.metrics` tables with correct column names.

### Test 2: Multi-Table Join
```bash
python finangpt.py query "Compare FAANG companies by revenue growth"
```

**Expected**: Should join `company.peers`, `financials.annual`, and use window functions.

### Test 3: Technical Analysis
```bash
python finangpt.py query "Find oversold stocks with RSI < 30"
```

**Expected**: Should query `technical.indicators` table with correct column `rsi_14`.

## Troubleshooting

### Model outputs prose instead of SQL

**Symptom**:
```
SQLExtractionError: Could not extract SQL from LLM response.
Response preview: Here's a quick snapshot...
```

**Solution**: Switch to a code-specialized model
```bash
ollama pull qwen2.5-coder:14b
# Update .env: MODEL_NAME=qwen2.5-coder:14b
```

### Model uses wrong table names

**Symptom**:
```
Query failed: Table stocks is not on the allow-list.
```

**Solution**: The model is hallucinating. Use `qwen3:14b` or `mistral-small3.2:latest`

### Model uses wrong column names

**Symptom**:
```
Unknown columns in SELECT: m.company_name
```

**Solution**: Model isn't reading schema properly. Use `qwen2.5-coder:14b` which better follows structured prompts.

## Advanced Configuration

### Model Parameters (config.yaml)

```yaml
ollama:
  url: http://localhost:11434
  model: qwen2.5-coder:14b
  timeout: 60
  max_retries: 3
  # Optional: Tune generation parameters
  temperature: 0.1        # Lower = more deterministic
  top_p: 0.9             # Nucleus sampling
  top_k: 40              # Top-k sampling
```

### Custom Model Prompt

For advanced users, you can create custom Ollama Modelfiles with system prompts:

```dockerfile
FROM qwen2.5-coder:14b

SYSTEM """You are a DuckDB SQL generator. Output ONLY SQL code in ```sql``` fences."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
```

Save as `Modelfile`, then:
```bash
ollama create finangpt-sql -f Modelfile
# Update .env: MODEL_NAME=finangpt-sql
```

## Recommended Models by Use Case

### For Production Use
- **qwen2.5-coder:14b**: Most reliable SQL generation
- **deepseek-coder-v2:16b**: Complex financial queries

### For Development/Testing
- **qwen3:14b**: Fast iteration, good accuracy
- **mistral-small3.2**: Reliable fallback

### For Low-Resource Systems
- **qwen2.5-coder:7b**: Smaller version, still good
- **codellama:7b**: Lightweight code model

### For Research/Experimentation
- **llama3.1:8b**: General purpose baseline
- **gemma2:9b**: Google's instruction-tuned model

## Conclusion

For best results with FinanGPT:

1. **Use a code-specialized model** (qwen2.5-coder, deepseek-coder)
2. **Allocate sufficient RAM** (16GB+ for 14B models)
3. **Test with sample queries** before production use
4. **Monitor query success rate** and adjust if needed

The difference between a good model (qwen2.5-coder) and a poor fit (gpt-oss) is **80%+ query success rate** vs **20-40%**.

---

**Need help?** Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) or open an issue on GitHub.
