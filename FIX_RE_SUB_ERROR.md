# Fix: re.sub() Missing Argument Error

## Problem

When running queries, the system would crash with:
```
TypeError: sub() missing 1 required positional argument: 'string'
```

## Root Cause

In `src/query_engine/query.py` line 1024, the `re.sub()` function was called with incorrect arguments:

**Before (INCORRECT)**:
```python
sql_no_comments = re.sub(r'/\*.*?\*/', sql_no_comments, flags=re.DOTALL)
#                         ^pattern    ^string (WRONG!)   ^flags
#                                     Missing: replacement argument
```

The correct signature for `re.sub()` is:
```python
re.sub(pattern, repl, string, count=0, flags=0)
```

## Fix Applied

**After (CORRECT)**:
```python
sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
#                         ^pattern    ^repl  ^string        ^flags
```

Added the missing empty string `''` as the replacement argument to remove multi-line SQL comments.

## Verification

1. **AST Analysis**: Scanned entire codebase for similar `re.sub()` errors - none found
2. **Manual Review**: Checked all `re.sub()` calls with `flags=` parameter - all correct
3. **Testing**: 
   - ✅ Query "What are the top 10 companies by market cap?" - SUCCESS
   - ✅ Returns correct results with chart generation
   - ✅ No more TypeError

## Impact

- **Files Modified**: 1 (`src/query_engine/query.py`)
- **Lines Changed**: 1 (line 1024)
- **Breaking Changes**: None
- **Side Effects**: None - this was a pure bug fix

## Related Code

The error was in the SQL comment removal logic that strips comments before validation:

```python
# Strip SQL comments first (LLM-generated comments are safe after extraction)
# Remove single-line comments (-- ...)
sql_no_comments = re.sub(r'--[^\n]*', '', sql)
# Remove multi-line comments (/* ... */) - THIS LINE WAS BROKEN
sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)  # FIXED
```

## Lessons Learned

When using `flags=` as a keyword argument in `re.sub()`, all three required positional arguments must still be provided:
1. `pattern` - the regex pattern to match
2. `repl` - the replacement string  
3. `string` - the string to search in

The `flags` parameter is always optional and should come last.
