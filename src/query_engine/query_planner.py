"""
Query Decomposition & Planning Module

Breaks complex multi-part queries into sequential execution steps.

Author: FinanGPT Team
"""

import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import duckdb
import pandas as pd


@dataclass
class QueryStep:
    """
    Represents a single step in a query execution plan.

    Attributes:
        step_number: Step number (1-indexed)
        description: Human-readable description
        sql: SQL query to execute
        depends_on: List of step numbers this depends on
        result_var: Variable name to store results
    """
    step_number: int
    description: str
    sql: str
    depends_on: List[int]
    result_var: str = "result"


@dataclass
class QueryPlan:
    """
    Represents a complete query execution plan.

    Attributes:
        original_query: User's original query
        steps: List of query steps
        is_complex: Whether query needed decomposition
    """
    original_query: str
    steps: List[QueryStep]
    is_complex: bool = True


class QueryPlanner:
    """
    Decomposes complex queries into sequential execution steps.

    Features:
    - Detects if query needs decomposition
    - Generates step-by-step execution plan
    - Executes plan with intermediate result passing
    - Provides progress feedback
    """

    def __init__(
        self,
        db_connection: duckdb.DuckDBPyConnection,
        llm_callable: Optional[Any] = None
    ):
        """
        Initialize QueryPlanner.

        Args:
            db_connection: DuckDB connection
            llm_callable: Function to call LLM for plan generation
        """
        self.conn = db_connection
        self.llm_callable = llm_callable
        self._init_complexity_patterns()

    def _init_complexity_patterns(self):
        """Initialize patterns that indicate query complexity."""
        self.complexity_indicators = [
            # Multiple conjunctions
            r'\band\b.*\band\b.*\band\b',  # 3+ ANDs
            # Multiple conditions with different tables
            r'(price|valuation|earnings|analyst|technical).*\band\b.*(price|valuation|earnings|analyst|technical)',
            # Nested comparisons
            r'(with|that have|where).*\band\b.*(with|that have|where)',
            # Multiple metrics
            r'(p/e|pe ratio|roe|rsi|earnings|revenue|price).*\band\b.*(p/e|pe ratio|roe|rsi|earnings|revenue|price).*\band\b',
        ]

    def needs_decomposition(self, query: str) -> bool:
        """
        Determine if query is complex enough to need decomposition.

        Args:
            query: User query

        Returns:
            True if query should be decomposed
        """
        query_lower = query.lower()

        # Check complexity indicators
        for pattern in self.complexity_indicators:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        # Count conjunctions
        and_count = len(re.findall(r'\band\b', query_lower))
        if and_count >= 3:
            return True

        # Count table/domain references
        domains = ["price", "valuation", "earnings", "analyst", "technical", "financial"]
        domain_count = sum(1 for domain in domains if domain in query_lower)
        if domain_count >= 2:
            return True

        return False

    def create_plan(self, query: str, system_prompt: str = "") -> QueryPlan:
        """
        Create execution plan for query.

        Args:
            query: User query
            system_prompt: System prompt with schema information

        Returns:
            QueryPlan object
        """
        # Check if decomposition needed
        if not self.needs_decomposition(query):
            return QueryPlan(
                original_query=query,
                steps=[],
                is_complex=False
            )

        # Generate plan using LLM
        if self.llm_callable:
            plan_steps = self._generate_plan_with_llm(query, system_prompt)
        else:
            # Fallback: rule-based decomposition
            plan_steps = self._generate_plan_rule_based(query)

        return QueryPlan(
            original_query=query,
            steps=plan_steps,
            is_complex=True
        )

    def _generate_plan_with_llm(self, query: str, system_prompt: str) -> List[QueryStep]:
        """Generate plan using LLM."""
        planning_prompt = f"""
You are a query planning assistant. Break this complex query into sequential steps.

User Query: "{query}"

Generate a JSON array of steps with this structure:
[
  {{
    "step_number": 1,
    "description": "Brief description",
    "sql": "SELECT ... FROM ...",
    "depends_on": [],
    "result_var": "result1"
  }}
]

Each step should:
1. Be executable independently
2. Store results in a temp table (CREATE TEMP TABLE result1 AS ...)
3. Reference previous results using result_var from depends_on steps
4. Build toward the final answer

Return ONLY the JSON array, no other text.
"""

        try:
            # Call LLM
            response = self.llm_callable(planning_prompt)

            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                steps_data = json.loads(json_match.group(0))
                steps = []
                for s in steps_data:
                    steps.append(QueryStep(
                        step_number=s["step_number"],
                        description=s["description"],
                        sql=s["sql"],
                        depends_on=s.get("depends_on", []),
                        result_var=s.get("result_var", f"result{s['step_number']}")
                    ))
                return steps
            else:
                # Fallback to rule-based
                return self._generate_plan_rule_based(query)

        except Exception as e:
            print(f"Error generating LLM plan: {e}")
            return self._generate_plan_rule_based(query)

    def _generate_plan_rule_based(self, query: str) -> List[QueryStep]:
        """Generate plan using simple rules (fallback)."""
        query_lower = query.lower()
        steps = []

        # Step 1: Identify main entity (ticker, sector, etc.)
        if "tech" in query_lower or "sector" in query_lower:
            steps.append(QueryStep(
                step_number=1,
                description="Filter companies by sector",
                sql="CREATE TEMP TABLE result1 AS SELECT ticker FROM company.metadata WHERE sector = 'Technology'",
                depends_on=[],
                result_var="result1"
            ))

        # Step 2: Apply financial filters
        if "p/e" in query_lower or "valuation" in query_lower:
            prev_step = len(steps)
            depends = [prev_step] if prev_step > 0 else []
            where_clause = " AND ticker IN (SELECT ticker FROM result1)" if depends else ""
            steps.append(QueryStep(
                step_number=len(steps) + 1,
                description="Filter by valuation metrics",
                sql=f"CREATE TEMP TABLE result{len(steps) + 1} AS SELECT ticker FROM valuation.metrics WHERE pe_ratio < 15{where_clause}",
                depends_on=depends,
                result_var=f"result{len(steps) + 1}"
            ))

        # Step 3: Apply analyst filters
        if "upgrade" in query_lower or "analyst" in query_lower:
            prev_step = len(steps)
            depends = [prev_step] if prev_step > 0 else []
            where_clause = f" AND ticker IN (SELECT ticker FROM result{prev_step})" if depends else ""
            steps.append(QueryStep(
                step_number=len(steps) + 1,
                description="Check analyst recommendations",
                sql=f"CREATE TEMP TABLE result{len(steps) + 1} AS SELECT ticker FROM analyst.recommendations WHERE to_grade = 'Buy'{where_clause}",
                depends_on=depends,
                result_var=f"result{len(steps) + 1}"
            ))

        # Final step: Get full details
        if steps:
            final_result_var = steps[-1].result_var
            steps.append(QueryStep(
                step_number=len(steps) + 1,
                description="Get final results with details",
                sql=f"SELECT * FROM company.metadata WHERE ticker IN (SELECT ticker FROM {final_result_var})",
                depends_on=[len(steps)],
                result_var="final_result"
            ))

        return steps

    def execute_plan(self, plan: QueryPlan, verbose: bool = True) -> pd.DataFrame:
        """
        Execute query plan step by step.

        Args:
            plan: QueryPlan to execute
            verbose: Print progress messages

        Returns:
            Final result DataFrame
        """
        if not plan.is_complex:
            raise ValueError("Plan does not require decomposition. Use regular query execution.")

        if verbose:
            print(f"\n📋 Query Plan ({len(plan.steps)} steps):")
            for step in plan.steps:
                deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
                print(f"  {step.step_number}. {step.description}{deps}")
            print("\nExecuting...\n")

        # Execute each step
        results = {}
        for step in plan.steps:
            if verbose:
                print(f"✓ Step {step.step_number}: {step.description}...")

            try:
                # Execute SQL
                result = self.conn.execute(step.sql).fetchdf()

                # Store result
                results[step.result_var] = result

                if verbose:
                    if "CREATE TEMP TABLE" in step.sql:
                        row_count = len(result) if hasattr(result, '__len__') else "N/A"
                        print(f"  → {row_count} rows")
                    else:
                        print(f"  → {len(result)} rows returned")

            except Exception as e:
                print(f"❌ Step {step.step_number} failed: {e}")
                raise

        # Return final result
        final_step = plan.steps[-1]
        return results.get(final_step.result_var, pd.DataFrame())

    def cleanup_temp_tables(self):
        """Clean up temporary tables created during plan execution."""
        try:
            # Get list of temp tables
            temp_tables = self.conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'temp'
            """).fetchall()

            for (table_name,) in temp_tables:
                if table_name.startswith("result"):
                    self.conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")

        except Exception as e:
            print(f"Warning: Could not clean up temp tables: {e}")


def format_query_plan(plan: QueryPlan) -> str:
    """
    Format query plan for display.

    Args:
        plan: QueryPlan object

    Returns:
        Formatted string
    """
    if not plan.is_complex:
        return "Query does not require decomposition."

    lines = []
    lines.append("\n📋 Query Execution Plan:")
    lines.append("=" * 80)
    lines.append(f"Original Query: {plan.original_query}")
    lines.append(f"Steps: {len(plan.steps)}")
    lines.append("=" * 80)

    for step in plan.steps:
        lines.append(f"\nStep {step.step_number}: {step.description}")
        if step.depends_on:
            lines.append(f"  Depends on: Steps {', '.join(map(str, step.depends_on))}")
        lines.append(f"  SQL: {step.sql[:100]}..." if len(step.sql) > 100 else f"  SQL: {step.sql}")

    lines.append("=" * 80)
    return "\n".join(lines)


def is_query_complex(query: str) -> bool:
    """
    Quick check if query is complex (convenience function).

    Args:
        query: User query

    Returns:
        True if query appears complex
    """
    planner = QueryPlanner(None, None)
    return planner.needs_decomposition(query)
