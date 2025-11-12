# FinanGPT - Complete Project Specification

## Executive Summary

FinanGPT is an enterprise-grade financial intelligence platform that combines comprehensive global data ingestion, multi-currency support, and conversational AI analysis. The platform enables users to ask complex financial questions in natural language and receive analytical insights backed by data - not just SQL query results.

### Core Capabilities
- ✅ Agent-based analysis pipeline for multi-step reasoning
- ✅ LLM analyzes actual data and generates insights, not just SQL queries
- ✅ Concurrent data ingestion with intelligent caching
- ✅ Streaming transformation for memory-efficient processing
- ✅ Multi-currency support with automatic FX conversion
- ✅ Comprehensive data schema (22 DuckDB tables, 13 MongoDB collections)
- ✅ Enterprise-grade security with SQL guardrails
- ✅ Clean layered architecture for testability and scalability
- ✅ Context-aware conversations with memory
- ✅ Automatic visualization generation

### Key Innovation: True Data Analysis

The platform's core innovation is an **analysis pipeline** where the LLM doesn't just generate queries - it:
1. **Plans** complex analytical questions by breaking them into steps
2. **Executes** multiple queries to gather relevant data
3. **Analyzes** the actual data to extract meaningful insights
4. **Synthesizes** natural language answers with supporting evidence
5. **Visualizes** trends and comparisons automatically
6. **Maintains** conversation context for follow-up questions

This approach transforms financial data analysis from a technical SQL exercise into a natural conversation with an AI analyst.

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ CLI         │  │ Chat         │  │ Future: REST API │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼──────────────────┼─────────────┘
          │                 │                  │
┌─────────┴─────────────────┴──────────────────┴─────────────┐
│                   APPLICATION LAYER                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Analysis Orchestrator (NEW)                  │ │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────┐ │ │
│  │  │Query Planner │→ │Data Retriever │→ │Synthesizer │ │ │
│  │  └──────────────┘  └───────────────┘  └────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Ingest Use Case│  │Transform Use   │  │Visualize Use │ │
│  │                │  │Case            │  │Case          │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────────────┐
│                   DOMAIN LAYER                                │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Entities   │  │ Value       │  │ Domain Services      │  │
│  │ (Company,  │  │ Objects     │  │ (Analysis Rules,     │  │
│  │ Financial, │  │ (Money,     │  │ Validation Logic)    │  │
│  │ etc.)      │  │ Date Range) │  │                      │  │
│  └────────────┘  └─────────────┘  └──────────────────────┘  │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────────────┐
│                 INFRASTRUCTURE LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ MongoDB     │  │ DuckDB      │  │ LLM Service         │  │
│  │ Repository  │  │ Repository  │  │ (Ollama Adapter)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ YFinance    │  │ Query Cache │  │ Visualization       │  │
│  │ Adapter     │  │ Repository  │  │ Service             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Core Principles

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Dependency Inversion**: High-level modules don't depend on low-level modules
3. **Testability**: Every component can be tested in isolation
4. **Explicit Dependencies**: No hidden state or global variables
5. **Type Safety**: Full type hints throughout codebase
6. **Agent-First Design**: LLM is an active analyst, not just a query generator

---

## 2. Detailed Architecture

### 2.1 Domain Layer (Pure Business Logic)

**Purpose**: Core business entities and rules with NO external dependencies

#### 2.1.1 Entities

```python
# src/finangpt/domain/entities/company.py
@dataclass(frozen=True)
class Company:
    """Immutable company entity"""
    ticker: str
    name: str
    sector: str | None
    industry: str | None
    currency: str
    country: str
    exchange: str
    
    def __post_init__(self):
        if not self.ticker:
            raise ValueError("Ticker cannot be empty")
        if len(self.ticker) > 10:
            raise ValueError("Ticker too long")

# src/finangpt/domain/entities/financial_statement.py
@dataclass(frozen=True)
class FinancialStatement:
    """Annual or quarterly financial statement"""
    ticker: str
    period_end_date: date
    statement_type: Literal["annual", "quarterly"]
    total_revenue: Decimal | None
    net_income: Decimal | None
    total_assets: Decimal | None
    shareholder_equity: Decimal | None
    operating_cash_flow: Decimal | None
    # ... other fields
    
    @property
    def net_margin(self) -> Decimal | None:
        """Calculate net margin if possible"""
        if self.total_revenue and self.net_income and self.total_revenue > 0:
            return (self.net_income / self.total_revenue) * 100
        return None

# src/finangpt/domain/entities/price_data.py
@dataclass(frozen=True)
class PriceData:
    """Daily price information"""
    ticker: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal | None

# src/finangpt/domain/entities/analysis_result.py
@dataclass(frozen=True)
class AnalysisResult:
    """Result of an analytical operation"""
    query_text: str
    insights: list["Insight"]
    answer_text: str
    supporting_data: pd.DataFrame | None
    visualization_recommendation: str | None
    confidence_score: float  # 0.0 to 1.0
    timestamp: datetime

@dataclass(frozen=True)
class Insight:
    """A single analytical finding"""
    finding: str
    significance: str
    data_points: list[str]
    category: Literal["trend", "comparison", "anomaly", "correlation", "summary"]
```

#### 2.1.2 Value Objects

```python
# src/finangpt/domain/value_objects/money.py
@dataclass(frozen=True)
class Money:
    """Immutable money value with currency"""
    amount: Decimal
    currency: str
    
    def convert_to(self, target_currency: str, fx_rate: Decimal) -> "Money":
        """Convert to another currency"""
        return Money(
            amount=self.amount * fx_rate,
            currency=target_currency
        )
    
    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add money in different currencies")
        return Money(self.amount + other.amount, self.currency)

# src/finangpt/domain/value_objects/date_range.py
@dataclass(frozen=True)
class DateRange:
    """Immutable date range"""
    start_date: date
    end_date: date
    
    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
    
    def contains(self, check_date: date) -> bool:
        return self.start_date <= check_date <= self.end_date
    
    def days(self) -> int:
        return (self.end_date - self.start_date).days

# src/finangpt/domain/value_objects/query_plan.py
@dataclass(frozen=True)
class QueryPlan:
    """Plan for executing a complex query"""
    original_question: str
    steps: list["QueryStep"]
    requires_visualization: bool
    estimated_complexity: Literal["simple", "moderate", "complex"]

@dataclass(frozen=True)
class QueryStep:
    """Single step in query execution"""
    step_number: int
    description: str
    sql_query: str
    dependencies: list[int]  # Which steps must complete first
    expected_columns: list[str]
```

#### 2.1.3 Domain Services

```python
# src/finangpt/domain/services/financial_analyzer.py
class FinancialAnalyzer:
    """Pure business logic for financial analysis"""
    
    @staticmethod
    def calculate_growth_rate(
        earlier: FinancialStatement, 
        later: FinancialStatement
    ) -> Decimal | None:
        """Calculate year-over-year growth rate"""
        if not earlier.total_revenue or not later.total_revenue:
            return None
        if earlier.total_revenue == 0:
            return None
        return ((later.total_revenue - earlier.total_revenue) / 
                earlier.total_revenue * 100)
    
    @staticmethod
    def is_healthy_balance_sheet(statement: FinancialStatement) -> bool:
        """Determine if balance sheet shows healthy metrics"""
        if not statement.total_assets or not statement.total_liabilities:
            return False
        
        debt_ratio = statement.total_liabilities / statement.total_assets
        return debt_ratio < Decimal("0.6")  # Less than 60% debt

# src/finangpt/domain/services/query_validator.py
class QueryValidator:
    """Domain rules for query validation"""
    
    DANGEROUS_PATTERNS = [
        r";\s*DROP",
        r";\s*DELETE",
        r";\s*INSERT",
        r";\s*UPDATE",
        r"--",
        r"/\*",
    ]
    
    ALLOWED_SCHEMAS = [
        "financials", "prices", "dividends", "splits", 
        "company", "ratios", "growth", "valuation", 
        "earnings", "analyst", "technical", "currency"
    ]
    
    @classmethod
    def validate_sql(cls, sql: str) -> tuple[bool, str | None]:
        """Validate SQL query for security"""
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                return False, f"Dangerous SQL pattern detected: {pattern}"
        
        # Check for write operations
        if any(op in sql.upper() for op in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]):
            return False, "Write operations not allowed"
        
        # Check for allowed schemas
        # ... validation logic
        
        return True, None
```

---

### 2.2 Application Layer (Use Cases & Orchestration)

**Purpose**: Coordinate domain logic and infrastructure to accomplish business goals

#### 2.2.1 Analysis Orchestrator (The Heart of the System)

```python
# src/finangpt/application/analysis/orchestrator.py
class AnalysisOrchestrator:
    """
    Main orchestrator for the analysis pipeline.
    This is the core of FinanGPT - it manages the entire
    analysis workflow from question to answer.
    """
    
    def __init__(
        self,
        query_planner: QueryPlanner,
        data_retriever: DataRetriever,
        result_analyzer: ResultAnalyzer,
        insight_synthesizer: InsightSynthesizer,
        viz_detector: VisualizationDetector,
        conversation_manager: ConversationManager,
    ):
        self._query_planner = query_planner
        self._data_retriever = data_retriever
        self._result_analyzer = result_analyzer
        self._insight_synthesizer = insight_synthesizer
        self._viz_detector = viz_detector
        self._conversation_manager = conversation_manager
    
    async def analyze_question(
        self, 
        question: str,
        conversation_id: str | None = None
    ) -> AnalysisResult:
        """
        Main entry point for analysis.
        
        Flow:
        1. Get conversation context if available
        2. Plan the query (break into steps)
        3. Execute each step and gather data
        4. Analyze the combined data
        5. Synthesize insights into natural language
        6. Detect visualization needs
        7. Update conversation context
        """
        
        # 1. Get context
        context = None
        if conversation_id:
            context = await self._conversation_manager.get_context(conversation_id)
        
        # 2. Plan query
        plan = await self._query_planner.create_plan(
            question=question,
            context=context
        )
        
        # 3. Execute steps and gather data
        all_data = {}
        for step in plan.steps:
            # Wait for dependencies
            dependency_data = {
                dep: all_data[dep] for dep in step.dependencies
            }
            
            # Execute this step
            step_data = await self._data_retriever.execute_step(
                step=step,
                dependencies=dependency_data
            )
            all_data[step.step_number] = step_data
        
        # 4. Analyze the data
        insights = await self._result_analyzer.analyze(
            question=question,
            query_plan=plan,
            data=all_data,
            context=context
        )
        
        # 5. Synthesize answer
        answer_text = await self._insight_synthesizer.synthesize(
            question=question,
            insights=insights,
            context=context
        )
        
        # 6. Detect visualization needs
        viz_rec = await self._viz_detector.detect(
            question=question,
            data=all_data,
            insights=insights
        )
        
        # 7. Create result
        result = AnalysisResult(
            query_text=question,
            insights=insights,
            answer_text=answer_text,
            supporting_data=self._combine_dataframes(all_data),
            visualization_recommendation=viz_rec,
            confidence_score=self._calculate_confidence(insights),
            timestamp=datetime.now()
        )
        
        # 8. Update conversation
        if conversation_id:
            await self._conversation_manager.add_exchange(
                conversation_id=conversation_id,
                question=question,
                result=result
            )
        
        return result
```

#### 2.2.2 Query Planner

```python
# src/finangpt/application/analysis/query_planner.py
class QueryPlanner:
    """
    Plans how to execute complex queries by breaking them into steps.
    Uses LLM with structured output.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        schema_provider: SchemaProvider,
    ):
        self._llm = llm_service
        self._schema = schema_provider
    
    async def create_plan(
        self, 
        question: str,
        context: ConversationContext | None
    ) -> QueryPlan:
        """
        Create a structured execution plan for the question.
        
        The LLM is prompted to output JSON with this structure:
        {
            "complexity": "simple|moderate|complex",
            "requires_visualization": true|false,
            "steps": [
                {
                    "step_number": 1,
                    "description": "Get revenue data for AAPL",
                    "sql_query": "SELECT ...",
                    "dependencies": [],
                    "expected_columns": ["ticker", "revenue", "date"]
                },
                ...
            ]
        }
        """
        
        # Get schema info
        schema_info = await self._schema.get_schema_description()
        
        # Build prompt
        prompt = self._build_planning_prompt(
            question=question,
            schema_info=schema_info,
            context=context
        )
        
        # Get structured output from LLM
        response = await self._llm.generate_structured(
            prompt=prompt,
            output_schema=QueryPlanSchema,
            temperature=0.1  # Low temperature for consistency
        )
        
        # Validate and return
        return self._validate_and_build_plan(question, response)
    
    def _build_planning_prompt(
        self, 
        question: str, 
        schema_info: str,
        context: ConversationContext | None
    ) -> str:
        """Build the LLM prompt for query planning"""
        
        prompt = f"""You are a financial data analyst planning how to answer a question.

QUESTION: {question}

AVAILABLE DATABASE SCHEMA:
{schema_info}

CONTEXT FROM CONVERSATION:
{context.summary if context else "No prior context"}

Your task:
1. Analyze the question complexity
2. Break it into logical steps (queries)
3. Identify dependencies between steps
4. Write SQL for each step
5. Determine if visualization would be helpful

Output a JSON plan following this exact structure:
{{
    "complexity": "simple|moderate|complex",
    "requires_visualization": true|false,
    "reasoning": "Brief explanation of your approach",
    "steps": [
        {{
            "step_number": 1,
            "description": "Clear description of what this step does",
            "sql_query": "SELECT ... (valid DuckDB SQL)",
            "dependencies": [list of step numbers this depends on],
            "expected_columns": ["col1", "col2"]
        }}
    ]
}}

Rules:
- For simple questions (single metric, single company): 1 step
- For comparisons: Usually 1-2 steps
- For complex analysis (trends, correlations, rankings): 2-5 steps
- Each step must be independently executable
- Dependencies must reference earlier steps only
- SQL must be valid DuckDB syntax
- Use only schemas listed in AVAILABLE DATABASE SCHEMA
"""
        return prompt
```

#### 2.2.3 Result Analyzer

```python
# src/finangpt/application/analysis/result_analyzer.py
class ResultAnalyzer:
    """
    Analyzes query results to extract insights.
    This is where the LLM actually LOOKS at the data.
    """
    
    def __init__(self, llm_service: LLMService):
        self._llm = llm_service
    
    async def analyze(
        self,
        question: str,
        query_plan: QueryPlan,
        data: dict[int, pd.DataFrame],
        context: ConversationContext | None
    ) -> list[Insight]:
        """
        Analyze the data and extract insights.
        
        The LLM sees the actual data values and extracts meaningful
        patterns, trends, and insights to answer the question.
        """
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(data)
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            question=question,
            plan=query_plan,
            data_summary=data_summary,
            context=context
        )
        
        # Get structured insights from LLM
        response = await self._llm.generate_structured(
            prompt=prompt,
            output_schema=InsightsSchema,
            temperature=0.3  # Slightly higher for creative analysis
        )
        
        return self._parse_insights(response)
    
    def _prepare_data_summary(self, data: dict[int, pd.DataFrame]) -> str:
        """
        Prepare a concise summary of data for LLM.
        Include: shape, columns, sample rows, statistics.
        """
        summaries = []
        
        for step_num, df in data.items():
            if len(df) == 0:
                summaries.append(f"Step {step_num}: No data returned")
                continue
            
            summary = f"""
Step {step_num}:
- Rows: {len(df)}
- Columns: {', '.join(df.columns.tolist())}
- Sample data (first 5 rows):
{df.head(5).to_string(index=False)}

- Summary statistics:
{df.describe().to_string()}
"""
            summaries.append(summary)
        
        return "\n\n".join(summaries)
    
    def _build_analysis_prompt(
        self,
        question: str,
        plan: QueryPlan,
        data_summary: str,
        context: ConversationContext | None
    ) -> str:
        """Build prompt for data analysis"""
        
        return f"""You are a financial analyst examining data to answer a question.

ORIGINAL QUESTION: {question}

QUERY PLAN EXECUTED:
{self._format_plan(plan)}

DATA RETRIEVED:
{data_summary}

CONVERSATION CONTEXT:
{context.summary if context else "No prior context"}

Your task:
Analyze the data above and extract meaningful insights. For each insight:
1. State the finding clearly
2. Explain why it's significant
3. Reference specific data points

Output JSON in this structure:
{{
    "insights": [
        {{
            "finding": "Clear statement of what you found",
            "significance": "Why this matters",
            "data_points": ["Specific values that support this"],
            "category": "trend|comparison|anomaly|correlation|summary"
        }}
    ],
    "data_quality_notes": "Any concerns about the data",
    "confidence": 0.0-1.0
}}

Focus on:
- Trends over time
- Comparisons between entities
- Unexpected values or anomalies
- Correlations or patterns
- Direct answers to the question
"""
```

#### 2.2.4 Insight Synthesizer

```python
# src/finangpt/application/analysis/insight_synthesizer.py
class InsightSynthesizer:
    """
    Synthesizes insights into natural language answers.
    This is the final step that produces the user-facing response.
    """
    
    def __init__(self, llm_service: LLMService):
        self._llm = llm_service
    
    async def synthesize(
        self,
        question: str,
        insights: list[Insight],
        context: ConversationContext | None
    ) -> str:
        """
        Synthesize insights into a coherent natural language answer.
        """
        
        prompt = self._build_synthesis_prompt(question, insights, context)
        
        response = await self._llm.generate_text(
            prompt=prompt,
            temperature=0.4,  # Balanced for natural language
            max_tokens=500
        )
        
        return response.strip()
    
    def _build_synthesis_prompt(
        self,
        question: str,
        insights: list[Insight],
        context: ConversationContext | None
    ) -> str:
        """Build prompt for answer synthesis"""
        
        insights_text = "\n\n".join([
            f"Insight {i+1} ({ins.category}):\n"
            f"- Finding: {ins.finding}\n"
            f"- Significance: {ins.significance}\n"
            f"- Data: {', '.join(ins.data_points)}"
            for i, ins in enumerate(insights)
        ])
        
        return f"""You are a financial analyst providing a clear answer to a client's question.

QUESTION: {question}

INSIGHTS FROM DATA ANALYSIS:
{insights_text}

CONVERSATION CONTEXT:
{context.summary if context else "First question in conversation"}

Your task:
Write a clear, concise answer that:
1. Directly answers the question
2. Incorporates the key insights
3. Uses specific numbers and facts
4. Maintains a professional but conversational tone
5. Is structured logically (most important points first)

Guidelines:
- Start with a direct answer
- Support with specific data points
- Mention any caveats or limitations
- Keep it under 300 words
- Use bullet points for lists (when appropriate)
- Don't just repeat the insights - synthesize them

Generate your answer:"""
```

#### 2.2.5 Conversation Manager

```python
# src/finangpt/application/conversation/conversation_manager.py
class ConversationManager:
    """Manages conversation state and context"""
    
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        context_builder: ContextBuilder
    ):
        self._repo = conversation_repo
        self._context_builder = context_builder
    
    async def get_context(self, conversation_id: str) -> ConversationContext:
        """Get relevant context from conversation history"""
        
        # Get recent exchanges
        history = await self._repo.get_history(
            conversation_id=conversation_id,
            limit=10  # Last 10 exchanges
        )
        
        # Build context summary
        context = await self._context_builder.build_context(history)
        return context
    
    async def add_exchange(
        self,
        conversation_id: str,
        question: str,
        result: AnalysisResult
    ):
        """Add a question-answer exchange to history"""
        
        exchange = ConversationExchange(
            conversation_id=conversation_id,
            question=question,
            answer=result.answer_text,
            insights=result.insights,
            timestamp=result.timestamp
        )
        
        await self._repo.save_exchange(exchange)

# src/finangpt/application/conversation/context_builder.py
class ContextBuilder:
    """Builds context summaries from conversation history"""
    
    def __init__(self, llm_service: LLMService):
        self._llm = llm_service
    
    async def build_context(
        self, 
        history: list[ConversationExchange]
    ) -> ConversationContext:
        """
        Build a context summary from conversation history.
        This helps with follow-up questions like "What about Microsoft?"
        """
        
        if not history:
            return ConversationContext(
                summary="No prior conversation",
                referenced_tickers=set(),
                referenced_metrics=set(),
                time_range=None
            )
        
        # Extract key information
        tickers = set()
        metrics = set()
        
        for exchange in history:
            # Extract tickers mentioned
            tickers.update(self._extract_tickers(exchange.question))
            tickers.update(self._extract_tickers(exchange.answer))
            
            # Extract metrics mentioned
            metrics.update(self._extract_metrics(exchange.question))
        
        # Build summary with LLM
        summary_prompt = f"""Summarize this conversation history in 2-3 sentences:

{self._format_history(history)}

Focus on:
- Main topics discussed
- Companies/tickers analyzed
- Time periods referenced
- Key conclusions reached"""
        
        summary = await self._llm.generate_text(
            prompt=summary_prompt,
            temperature=0.2,
            max_tokens=150
        )
        
        return ConversationContext(
            summary=summary,
            referenced_tickers=tickers,
            referenced_metrics=metrics,
            time_range=self._infer_time_range(history)
        )
```

---

### 2.3 Infrastructure Layer

#### 2.3.1 LLM Service

```python
# src/finangpt/infrastructure/llm/ollama_service.py
class OllamaLLMService(LLMService):
    """Ollama implementation of LLM service"""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60
    ):
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        temperature: float = 0.1
    ) -> dict:
        """
        Generate structured output conforming to a Pydantic schema.
        
        Uses JSON mode to ensure valid structured output.
        """
        
        # Add schema to prompt
        schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
        full_prompt = f"""{prompt}

OUTPUT SCHEMA (you must return valid JSON matching this exact structure):
{schema_str}"""
        
        # Call Ollama with JSON mode
        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": full_prompt,
                "format": "json",  # Force JSON output
                "temperature": temperature,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise LLMError(f"Ollama API error: {response.text}")
        
        result = response.json()
        output_text = result["response"]
        
        # Parse and validate against schema
        try:
            parsed = json.loads(output_text)
            validated = output_schema(**parsed)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            # Retry once with error feedback
            return await self._retry_with_feedback(
                prompt, output_schema, temperature, str(e), output_text
            )
    
    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 500
    ) -> str:
        """Generate free-form text"""
        
        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
        )
        
        if response.status_code != 200:
            raise LLMError(f"Ollama API error: {response.text}")
        
        return response.json()["response"]
```

#### 2.3.2 Repository Implementations

```python
# src/finangpt/infrastructure/persistence/duckdb_repository.py
class DuckDBFinancialRepository(FinancialRepository):
    """DuckDB implementation for financial data access"""
    
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self._conn = connection
    
    async def get_financials(
        self,
        ticker: str,
        statement_type: Literal["annual", "quarterly"],
        start_date: date | None = None,
        end_date: date | None = None
    ) -> list[FinancialStatement]:
        """Get financial statements"""
        
        table = "financials.annual" if statement_type == "annual" else "financials.quarterly"
        
        query = f"SELECT * FROM {table} WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND period_end_date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND period_end_date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY period_end_date"
        
        df = self._conn.execute(query, params).df()
        
        return [self._row_to_entity(row) for _, row in df.iterrows()]
    
    async def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL query"""
        # Validation happens at application layer
        return self._conn.execute(sql).df()

# src/finangpt/infrastructure/persistence/mongodb_repository.py
class MongoDBRawDataRepository(RawDataRepository):
    """MongoDB implementation for raw data storage"""
    
    def __init__(self, database: Database):
        self._db = database
    
    async def save_raw_financials(
        self,
        ticker: str,
        data: dict,
        statement_type: Literal["annual", "quarterly"]
    ):
        """Save raw financial data"""
        
        collection = "raw_annual" if statement_type == "annual" else "raw_quarterly"
        
        document = {
            "ticker": ticker,
            "fetched_at": datetime.now(),
            "data": data
        }
        
        await self._db[collection].update_one(
            {"ticker": ticker},
            {"$set": document},
            upsert=True
        )
```

---

### 2.4 Interface Layer

```python
# src/finangpt/interface/cli/chat.py
class ChatCLI:
    """CLI for interactive chat"""
    
    def __init__(self, orchestrator: AnalysisOrchestrator):
        self._orchestrator = orchestrator
        self._conversation_id = str(uuid.uuid4())
    
    async def run(self):
        """Run interactive chat loop"""
        
        print("Welcome to FinanGPT!")
        print("Ask me questions about financial data.")
        print("Type 'exit' to quit, 'help' for commands.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() == "exit":
                    break
                
                if question.lower() == "help":
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                # Show thinking indicator
                print("\nAnalyzing...", end="", flush=True)
                
                # Get answer
                result = await self._orchestrator.analyze_question(
                    question=question,
                    conversation_id=self._conversation_id
                )
                
                # Clear thinking indicator
                print("\r              \r", end="")
                
                # Display answer
                print(f"Assistant: {result.answer_text}\n")
                
                # Show insights if verbose mode
                if self._verbose:
                    self._display_insights(result.insights)
                
                # Generate visualization if recommended
                if result.visualization_recommendation:
                    self._generate_visualization(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
```

---

## 3. Data Schema

### 3.1 DuckDB Schema (22 Tables)

```sql
-- Financial data
CREATE SCHEMA financials;
CREATE TABLE financials.annual (...);
CREATE TABLE financials.quarterly (...);

-- Price data  
CREATE SCHEMA prices;
CREATE TABLE prices.daily (...);

-- Corporate actions
CREATE SCHEMA dividends;
CREATE TABLE dividends.history (...);

CREATE SCHEMA splits;
CREATE TABLE splits.history (...);

-- Company information
CREATE SCHEMA company;
CREATE TABLE company.metadata (...);
CREATE TABLE company.peers (...);

-- Derived metrics
CREATE SCHEMA ratios;
CREATE TABLE ratios.financial (...);

CREATE SCHEMA growth;
CREATE VIEW growth.annual (...);

-- Valuation
CREATE SCHEMA valuation;
CREATE TABLE valuation.metrics (...);
CREATE TABLE valuation.metrics_multicurrency (...);

-- Earnings
CREATE SCHEMA earnings;
CREATE TABLE earnings.history (...);
CREATE TABLE earnings.calendar (...);

-- Analyst data
CREATE SCHEMA analyst;
CREATE TABLE analyst.recommendations (...);
CREATE TABLE analyst.price_targets (...);
CREATE TABLE analyst.consensus (...);
CREATE TABLE analyst.growth_estimates (...);

-- Technical indicators
CREATE SCHEMA technical;
CREATE TABLE technical.indicators (...);

-- Currency
CREATE SCHEMA currency;
CREATE TABLE currency.exchange_rates (...);

-- User data
CREATE SCHEMA user;
CREATE TABLE user.portfolios (...);
CREATE TABLE user.watchlists (...);
```

### 3.2 MongoDB Collections (13 Collections)

MongoDB stores raw data from external sources before transformation:

- raw_annual
- raw_quarterly  
- stock_prices_daily
- dividends_history
- splits_history
- company_metadata
- ingestion_metadata
- earnings_history
- earnings_calendar
- analyst_recommendations
- price_targets
- analyst_consensus
- growth_estimates

---

## 4. Project Structure

```
finangpt/
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt
│
├── config/
│   ├── config.yaml                 # Main configuration
│   ├── config.dev.yaml             # Development overrides
│   └── config.prod.yaml            # Production overrides
│
├── src/
│   └── finangpt/
│       ├── __init__.py
│       │
│       ├── domain/                 # Pure business logic (no dependencies)
│       │   ├── __init__.py
│       │   ├── entities/
│       │   │   ├── __init__.py
│       │   │   ├── company.py
│       │   │   ├── financial_statement.py
│       │   │   ├── price_data.py
│       │   │   └── analysis_result.py
│       │   ├── value_objects/
│       │   │   ├── __init__.py
│       │   │   ├── money.py
│       │   │   ├── date_range.py
│       │   │   └── query_plan.py
│       │   ├── services/
│       │   │   ├── __init__.py
│       │   │   ├── financial_analyzer.py
│       │   │   └── query_validator.py
│       │   └── exceptions.py
│       │
│       ├── application/            # Use cases & orchestration
│       │   ├── __init__.py
│       │   ├── analysis/
│       │   │   ├── __init__.py
│       │   │   ├── orchestrator.py
│       │   │   ├── query_planner.py
│       │   │   ├── data_retriever.py
│       │   │   ├── result_analyzer.py
│       │   │   ├── insight_synthesizer.py
│       │   │   └── visualization_detector.py
│       │   ├── conversation/
│       │   │   ├── __init__.py
│       │   │   ├── conversation_manager.py
│       │   │   └── context_builder.py
│       │   ├── ingestion/
│       │   │   ├── __init__.py
│       │   │   ├── ingest_use_case.py
│       │   │   └── validators.py
│       │   ├── transformation/
│       │   │   ├── __init__.py
│       │   │   └── transform_use_case.py
│       │   └── ports/              # Interfaces for infrastructure
│       │       ├── __init__.py
│       │       ├── llm_service.py
│       │       ├── repositories.py
│       │       └── cache_service.py
│       │
│       ├── infrastructure/         # External services & databases
│       │   ├── __init__.py
│       │   ├── llm/
│       │   │   ├── __init__.py
│       │   │   ├── ollama_service.py
│       │   │   └── schemas.py      # Pydantic schemas for LLM I/O
│       │   ├── persistence/
│       │   │   ├── __init__.py
│       │   │   ├── duckdb_repository.py
│       │   │   ├── mongodb_repository.py
│       │   │   ├── cache_repository.py
│       │   │   └── conversation_repository.py
│       │   ├── external/
│       │   │   ├── __init__.py
│       │   │   └── yfinance_adapter.py
│       │   └── visualization/
│       │       ├── __init__.py
│       │       └── matplotlib_service.py
│       │
│       ├── interface/              # User interfaces
│       │   ├── __init__.py
│       │   ├── cli/
│       │   │   ├── __init__.py
│       │   │   ├── chat.py
│       │   │   ├── query.py
│       │   │   ├── ingest.py
│       │   │   └── status.py
│       │   └── api/                # Future: REST API
│       │       └── __init__.py
│       │
│       └── shared/                 # Common utilities
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── dependency_injection.py
│           ├── types.py
│           └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/                       # Fast, isolated tests
│   │   ├── domain/
│   │   │   ├── test_entities.py
│   │   │   ├── test_value_objects.py
│   │   │   └── test_domain_services.py
│   │   ├── application/
│   │   │   ├── test_orchestrator.py
│   │   │   ├── test_query_planner.py
│   │   │   └── test_analyzers.py
│   │   └── shared/
│   │       └── test_utils.py
│   ├── integration/                # Tests with real services
│   │   ├── test_llm_integration.py
│   │   ├── test_database_integration.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_data.py
│       ├── mock_llm_responses.py
│       └── test_databases.py
│
├── scripts/                        # Utility scripts
│   ├── setup_databases.py
│   ├── migrate_v1_to_v2.py
│   └── benchmark.py
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    ├── migration_guide.md
    └── development_guide.md
```

---

## 5. Technology Stack

### 5.1 Core Technologies
- **Python**: 3.13+ (leveraging performance improvements and modern syntax)
  - *Why 3.13?* Released October 2024, stable for over a year, includes:
    - JIT compiler for improved performance
    - Better error messages
    - Enhanced type system features
    - Mature library support across the ecosystem
- **MongoDB**: Raw data storage
- **DuckDB**: High-performance analytics database
- **Ollama**: Local LLM inference
- **yfinance**: Market data ingestion

### 5.2 Key Dependencies

```toml
# pyproject.toml
[project]
name = "finangpt"
version = "2.0.0"
requires-python = ">=3.13"

dependencies = [
    # Core dependencies
    "pymongo>=4.6.0",
    "duckdb>=0.9.0",
    "pandas>=2.1.0",
    "yfinance>=0.2.28",
    "httpx>=0.25.0",           # For Ollama API
    "matplotlib>=3.8.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.66.0",
    
    # Architecture support
    "pydantic>=2.5.0",         # Data validation, structured LLM outputs
    "pydantic-settings>=2.1.0", # Configuration management
    "dependency-injector>=4.41.0", # DI container
    "structlog>=23.2.0",       # Structured logging
    "tenacity>=8.2.3",         # Retry logic
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "black>=23.12.0",
    "ruff>=0.1.9",
    "mypy>=1.7.0",
    "pre-commit>=3.6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (requires services)",
    "slow: Slow tests",
]

[tool.black]
line-length = 100
target-version = ["py313"]

[tool.ruff]
line-length = 100
target-version = "py313"
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "DTZ", "T10", "EM", "FA", "ISC", "ICN", "PIE", "PT", "RSE", "RET", "SIM", "TID", "ARG", "PTH", "PL", "TRY", "RUF"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

## 6. Configuration System

```yaml
# config/config.yaml
database:
  mongodb:
    uri: ${MONGO_URI:mongodb://localhost:27017/financial_data}
    pool_size: 10
    timeout_ms: 5000
  
  duckdb:
    path: ${DUCKDB_PATH:./data/financial_data.duckdb}
    readonly: false
    memory_limit: "2GB"

llm:
  provider: "ollama"  # Future: "openai", "anthropic"
  
  ollama:
    url: ${OLLAMA_URL:http://localhost:11434}
    model: ${MODEL_NAME:phi4:latest}
    timeout: 60
    max_retries: 3
  
  generation:
    planning_temperature: 0.1      # Low for consistency
    analysis_temperature: 0.3      # Moderate for insights
    synthesis_temperature: 0.4     # Balanced for natural language
    max_context_tokens: 4000
    max_output_tokens: 1000

ingestion:
  # Performance settings
  max_workers: 10
  worker_timeout: 120
  max_tickers_per_batch: 500
  price_lookback_days: 365
  auto_refresh_threshold_days: 7
  
  market_restrictions:
    mode: "global"  # "global", "us_only", "eu_only", "custom"
    exclude_etfs: true
    exclude_mutualfunds: true

transformation:
  # Streaming configuration
  chunk_size: 1000
  max_memory_mb: 2048
  enable_streaming: true
  run_integrity_checks: true

analysis:
  # Query execution settings
  max_query_steps: 10
  max_data_rows_per_step: 10000
  enable_query_caching: true
  cache_ttl_seconds: 300
  
  query_planning:
    max_retries: 2
    enable_validation: true
  
  result_analysis:
    max_insights: 10
    min_confidence_threshold: 0.5
  
  conversation:
    max_history_length: 10
    context_summary_enabled: true

visualization:
  enable: true
  output_dir: "./data/charts"
  dpi: 300
  default_style: "seaborn-v0_8"
  max_chart_retention: 1000

logging:
  level: ${LOG_LEVEL:INFO}
  format: "json"  # or "text"
  directory: "./data/logs"
  max_file_size_mb: 10
  backup_count: 5
  
  structured:
    enable: true
    include_context: true

monitoring:
  enable_metrics: false  # Future: Prometheus
  metrics_port: 9090
```

```python
# src/finangpt/shared/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MongoDBConfig(BaseSettings):
    uri: str = Field(default="mongodb://localhost:27017/financial_data")
    pool_size: int = 10
    timeout_ms: int = 5000

class DuckDBConfig(BaseSettings):
    path: str = Field(default="./data/financial_data.duckdb")
    readonly: bool = False
    memory_limit: str = "2GB"

class OllamaConfig(BaseSettings):
    url: str = Field(default="http://localhost:11434")
    model: str = Field(default="phi4:latest")
    timeout: int = 60
    max_retries: int = 3

class LLMGenerationConfig(BaseSettings):
    planning_temperature: float = 0.1
    analysis_temperature: float = 0.3
    synthesis_temperature: float = 0.4
    max_context_tokens: int = 4000
    max_output_tokens: int = 1000

class LLMConfig(BaseSettings):
    provider: str = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    generation: LLMGenerationConfig = Field(default_factory=LLMGenerationConfig)

class AnalysisConfig(BaseSettings):
    max_query_steps: int = 10
    max_data_rows_per_step: int = 10000
    enable_query_caching: bool = True
    cache_ttl_seconds: int = 300

class AppConfig(BaseSettings):
    """Main application configuration"""
    
    model_config = SettingsConfigDict(
        yaml_file="config/config.yaml",
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )
    
    database: DatabaseConfig
    llm: LLMConfig
    ingestion: IngestionConfig
    transformation: TransformationConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    logging: LoggingConfig

# Load configuration
def load_config() -> AppConfig:
    return AppConfig()
```

---

## 7. Dependency Injection

```python
# src/finangpt/shared/dependency_injection.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Configuration
    config = providers.Singleton(load_config)
    
    # Infrastructure - Databases
    mongodb_client = providers.Singleton(
        pymongo.MongoClient,
        config.provided.database.mongodb.uri,
        maxPoolSize=config.provided.database.mongodb.pool_size
    )
    
    mongodb_database = providers.Singleton(
        lambda client, config: client[config.database.mongodb.database_name],
        mongodb_client,
        config
    )
    
    duckdb_connection = providers.Singleton(
        duckdb.connect,
        config.provided.database.duckdb.path
    )
    
    # Infrastructure - LLM
    llm_service = providers.Factory(
        OllamaLLMService,
        base_url=config.provided.llm.ollama.url,
        model=config.provided.llm.ollama.model,
        timeout=config.provided.llm.ollama.timeout
    )
    
    # Infrastructure - Repositories
    financial_repository = providers.Factory(
        DuckDBFinancialRepository,
        connection=duckdb_connection
    )
    
    raw_data_repository = providers.Factory(
        MongoDBRawDataRepository,
        database=mongodb_database
    )
    
    conversation_repository = providers.Factory(
        SQLiteConversationRepository,
        path=config.provided.database.conversation.path
    )
    
    cache_repository = providers.Factory(
        RedisCacheRepository,
        url=config.provided.cache.redis_url
    )
    
    # Application - Analysis Components
    schema_provider = providers.Factory(
        DuckDBSchemaProvider,
        connection=duckdb_connection
    )
    
    query_planner = providers.Factory(
        QueryPlanner,
        llm_service=llm_service,
        schema_provider=schema_provider
    )
    
    data_retriever = providers.Factory(
        DataRetriever,
        repository=financial_repository,
        cache_repository=cache_repository,
        config=config.provided.analysis
    )
    
    result_analyzer = providers.Factory(
        ResultAnalyzer,
        llm_service=llm_service
    )
    
    insight_synthesizer = providers.Factory(
        InsightSynthesizer,
        llm_service=llm_service
    )
    
    visualization_detector = providers.Factory(
        VisualizationDetector,
        llm_service=llm_service
    )
    
    # Application - Conversation
    context_builder = providers.Factory(
        ContextBuilder,
        llm_service=llm_service
    )
    
    conversation_manager = providers.Factory(
        ConversationManager,
        conversation_repo=conversation_repository,
        context_builder=context_builder
    )
    
    # Application - Main Orchestrator
    analysis_orchestrator = providers.Factory(
        AnalysisOrchestrator,
        query_planner=query_planner,
        data_retriever=data_retriever,
        result_analyzer=result_analyzer,
        insight_synthesizer=insight_synthesizer,
        viz_detector=visualization_detector,
        conversation_manager=conversation_manager
    )
    
    # Interface - CLI
    chat_cli = providers.Factory(
        ChatCLI,
        orchestrator=analysis_orchestrator
    )

# Usage
container = Container()
chat = container.chat_cli()
await chat.run()
```

---

## 8. Testing Strategy

### 8.1 Test Pyramid

```
              ┌─────────────┐
              │   E2E (5%)  │  Full system tests
              └─────────────┘
           ┌──────────────────┐
           │Integration (20%) │  Component interaction
           └──────────────────┘
        ┌────────────────────────┐
        │    Unit Tests (75%)    │  Isolated logic
        └────────────────────────┘
```

### 8.2 Unit Tests

```python
# tests/unit/domain/test_value_objects.py
def test_money_cannot_add_different_currencies():
    usd = Money(Decimal("100"), "USD")
    eur = Money(Decimal("100"), "EUR")
    
    with pytest.raises(ValueError, match="different currencies"):
        _ = usd + eur

def test_money_conversion():
    usd = Money(Decimal("100"), "USD")
    eur = usd.convert_to("EUR", Decimal("0.85"))
    
    assert eur.amount == Decimal("85")
    assert eur.currency == "EUR"

# tests/unit/application/test_query_planner.py
@pytest.mark.asyncio
async def test_query_planner_simple_question():
    # Arrange
    mock_llm = Mock(spec=LLMService)
    mock_llm.generate_structured.return_value = {
        "complexity": "simple",
        "requires_visualization": False,
        "steps": [{
            "step_number": 1,
            "description": "Get AAPL revenue",
            "sql_query": "SELECT ticker, totalRevenue FROM financials.annual WHERE ticker = 'AAPL'",
            "dependencies": [],
            "expected_columns": ["ticker", "totalRevenue"]
        }]
    }
    
    schema_provider = Mock(spec=SchemaProvider)
    schema_provider.get_schema_description.return_value = "..."
    
    planner = QueryPlanner(mock_llm, schema_provider)
    
    # Act
    plan = await planner.create_plan("What is Apple's revenue?", None)
    
    # Assert
    assert plan.estimated_complexity == "simple"
    assert len(plan.steps) == 1
    assert plan.steps[0].sql_query.startswith("SELECT")
    assert "AAPL" in plan.steps[0].sql_query

# tests/unit/application/test_orchestrator.py
@pytest.mark.asyncio
async def test_orchestrator_full_flow():
    # Arrange
    mock_planner = Mock(spec=QueryPlanner)
    mock_planner.create_plan.return_value = QueryPlan(
        original_question="Test question",
        steps=[QueryStep(...)],
        requires_visualization=False,
        estimated_complexity="simple"
    )
    
    mock_retriever = Mock(spec=DataRetriever)
    mock_retriever.execute_step.return_value = pd.DataFrame({"revenue": [100, 200]})
    
    mock_analyzer = Mock(spec=ResultAnalyzer)
    mock_analyzer.analyze.return_value = [
        Insight(finding="Revenue increased", ...)
    ]
    
    mock_synthesizer = Mock(spec=InsightSynthesizer)
    mock_synthesizer.synthesize.return_value = "Apple's revenue is strong."
    
    # ... setup other mocks
    
    orchestrator = AnalysisOrchestrator(
        query_planner=mock_planner,
        data_retriever=mock_retriever,
        # ... other dependencies
    )
    
    # Act
    result = await orchestrator.analyze_question("Test question")
    
    # Assert
    assert result.answer_text == "Apple's revenue is strong."
    assert len(result.insights) == 1
    mock_planner.create_plan.assert_called_once()
    mock_retriever.execute_step.assert_called()
```

### 8.3 Integration Tests

```python
# tests/integration/test_llm_integration.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_planner_with_real_llm(ollama_service):
    """Test query planner with actual Ollama"""
    
    schema_provider = DuckDBSchemaProvider(test_duckdb_connection)
    planner = QueryPlanner(ollama_service, schema_provider)
    
    plan = await planner.create_plan(
        "Compare AAPL and MSFT revenue for 2023",
        context=None
    )
    
    assert plan.estimated_complexity in ["simple", "moderate", "complex"]
    assert len(plan.steps) >= 1
    assert all(step.sql_query for step in plan.steps)

# tests/integration/test_database_integration.py
@pytest.mark.integration
def test_financial_repository_get_financials(test_duckdb):
    """Test repository with real DuckDB"""
    
    # Setup test data
    test_duckdb.execute("""
        INSERT INTO financials.annual VALUES
        ('AAPL', '2023-09-30', 'annual', 394328000000, ...)
    """)
    
    repo = DuckDBFinancialRepository(test_duckdb)
    
    results = await repo.get_financials(
        ticker="AAPL",
        statement_type="annual"
    )
    
    assert len(results) == 1
    assert results[0].ticker == "AAPL"
    assert results[0].total_revenue == Decimal("394328000000")

# tests/integration/test_end_to_end.py
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_analysis_pipeline(container):
    """End-to-end test of complete analysis pipeline"""
    
    orchestrator = container.analysis_orchestrator()
    
    result = await orchestrator.analyze_question(
        "What is Apple's revenue for 2023?"
    )
    
    assert result.answer_text
    assert "Apple" in result.answer_text or "AAPL" in result.answer_text
    assert len(result.insights) > 0
    assert result.confidence_score > 0.5
```

### 8.4 Test Fixtures

```python
# tests/conftest.py
import pytest
from finangpt.shared.dependency_injection import Container

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "database": {
            "mongodb": {"uri": "mongodb://localhost:27017/test_financial_data"},
            "duckdb": {"path": ":memory:"}
        },
        "llm": {
            "ollama": {
                "url": "http://localhost:11434",
                "model": "phi4:latest"
            }
        }
    }

@pytest.fixture(scope="function")
def test_duckdb():
    """In-memory DuckDB for testing"""
    conn = duckdb.connect(":memory:")
    
    # Create test schema
    conn.execute("""
        CREATE SCHEMA financials;
        CREATE TABLE financials.annual (
            ticker VARCHAR,
            period_end_date DATE,
            ...
        );
    """)
    
    yield conn
    conn.close()

@pytest.fixture(scope="function")
def mock_llm_service():
    """Mock LLM service"""
    service = Mock(spec=LLMService)
    
    # Default responses
    service.generate_structured.return_value = {
        "steps": [{"step_number": 1, ...}]
    }
    service.generate_text.return_value = "Test answer"
    
    return service

@pytest.fixture(scope="session")
def ollama_service(test_config):
    """Real Ollama service for integration tests"""
    return OllamaLLMService(
        base_url=test_config["llm"]["ollama"]["url"],
        model=test_config["llm"]["ollama"]["model"]
    )

@pytest.fixture(scope="function")
def container(test_config, test_duckdb, mock_llm_service):
    """Dependency injection container for tests"""
    container = Container()
    container.config.override(test_config)
    container.duckdb_connection.override(test_duckdb)
    container.llm_service.override(mock_llm_service)
    return container
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Set up project structure and domain layer

- [ ] Create project structure
- [ ] Set up build system (pyproject.toml, setup.py)
- [ ] Implement domain entities
- [ ] Implement value objects
- [ ] Implement domain services
- [ ] Write unit tests for domain layer (100% coverage)
- [ ] Set up CI/CD pipeline

**Deliverable**: Clean domain layer with full test coverage

### Phase 2: Infrastructure (Week 2-3)
**Goal**: Implement infrastructure adapters

- [ ] Implement DuckDB repository
- [ ] Implement MongoDB repository
- [ ] Implement LLM service (Ollama adapter)
- [ ] Implement cache repository
- [ ] Implement conversation repository (SQLite)
- [ ] Write integration tests for each adapter
- [ ] Set up test fixtures and mocks

**Deliverable**: Working infrastructure layer with integration tests

### Phase 3: Analysis Pipeline (Week 3-5)
**Goal**: Build the core analysis orchestration

- [ ] Implement Query Planner with LLM structured outputs
- [ ] Implement Data Retriever
- [ ] Implement Result Analyzer
- [ ] Implement Insight Synthesizer
- [ ] Implement Visualization Detector
- [ ] Implement Analysis Orchestrator
- [ ] Write comprehensive unit tests
- [ ] Write integration tests for pipeline
- [ ] Test with example questions from spec

**Deliverable**: Working end-to-end analysis pipeline

### Phase 4: Conversation Management (Week 5-6)
**Goal**: Add conversation context and history

- [ ] Implement Conversation Manager
- [ ] Implement Context Builder
- [ ] Add conversation persistence
- [ ] Test multi-turn conversations
- [ ] Optimize context summarization

**Deliverable**: Context-aware conversational analysis

### Phase 5: Ingestion & Transformation (Week 6-7)
**Goal**: Implement data pipeline

- [ ] Implement concurrent ingestion logic
- [ ] Implement streaming transformation
- [ ] Integrate with architecture (repositories, DI)
- [ ] Implement data validation and filtering
- [ ] Implement FX rate caching
- [ ] Test with real data ingestion
- [ ] Verify data integrity

**Deliverable**: Working data pipeline

### Phase 6: CLI Interface (Week 7-8)
**Goal**: Build user interface

- [ ] Implement chat CLI
- [ ] Implement query CLI
- [ ] Implement status CLI
- [ ] Implement ingest CLI
- [ ] Add visualization rendering
- [ ] Add error handling and user feedback
- [ ] Test user workflows

**Deliverable**: Full-featured CLI

### Phase 7: Testing & Optimization (Week 8-10)
**Goal**: Comprehensive testing and performance optimization

- [ ] Achieve 90%+ test coverage
- [ ] Add end-to-end tests for all example questions
- [ ] Performance testing and optimization
- [ ] Memory profiling and optimization
- [ ] Load testing (concurrent queries)
- [ ] LLM prompt optimization
- [ ] Cache tuning

**Deliverable**: Production-ready, well-tested system

### Phase 8: Documentation (Week 10)
**Goal**: Complete documentation

- [ ] API reference documentation
- [ ] Architecture documentation
- [ ] User guide
- [ ] Development guide
- [ ] Deployment guide
- [ ] Example notebooks

**Deliverable**: Comprehensive documentation

### Phase 9: Deployment (Week 11)
**Goal**: Production deployment

- [ ] Set up production environment
- [ ] Configure monitoring and logging
- [ ] Deploy to production
- [ ] Performance tuning
- [ ] User training

**Deliverable**: Production-ready system

---

## 10. Architectural Benefits

### Design Philosophy

FinanGPT's architecture is built on several key principles that ensure scalability, maintainability, and quality:

| Aspect | Implementation | Benefit |
|--------|---------------|---------|
| **Architecture** | Clean layered architecture with dependency inversion | Easy to test, modify, and extend without breaking existing code |
| **LLM Integration** | Full analysis pipeline with structured outputs | LLM acts as an analyst, not just a query generator |
| **Data Flow** | Question → Plan → Execute → Analyze → Synthesize → Display | Transparent, debuggable multi-step reasoning |
| **Context Awareness** | Conversation manager with context summaries | Natural follow-up questions and coherent discussions |
| **Testing Strategy** | Dependency injection with comprehensive test suite | Fast iteration with confidence, 90%+ coverage |
| **Extensibility** | Protocol-based interfaces and strategy patterns | New features added without modifying core logic |
| **Observability** | Structured logging at each pipeline stage | Complete visibility into decision-making process |
| **Output Quality** | Natural language insights backed by data | Actionable analysis, not just raw numbers |
| **Query Complexity** | Automated query decomposition | Handles sophisticated analytical questions |
| **Code Organization** | Clear separation of concerns across layers | Maintainable, scalable codebase |

### Why This Matters

**Testability**: Every component can be tested in isolation. Mock dependencies enable fast unit tests without external services. Integration tests validate real-world behavior.

**Maintainability**: Changes to infrastructure (switching databases, LLM providers) don't affect business logic. New analysis strategies can be added without touching existing code.

**Scalability**: The architecture supports future enhancements like REST APIs, web UIs, and ML integrations without major refactoring.

**Quality**: Structured outputs from LLM ensure consistent, parseable responses. Multiple validation layers catch errors early.

**Developer Experience**: Clear module boundaries and type hints enable productive development. New team members can understand and contribute quickly.

---

## 11. Success Criteria

### Functional Requirements ✅

1. **Basic Queries**: System answers simple factual questions (e.g., "What is AAPL's revenue?") with natural language
2. **Complex Analysis**: System handles multi-step analytical questions from the example list
3. **Conversational**: System maintains context across multiple turns in a conversation
4. **Visualization**: System automatically generates appropriate charts when needed
5. **Multi-Currency**: System correctly handles international stocks and FX conversions
6. **Data Integrity**: All data pipeline features work correctly with validated outputs

### Non-Functional Requirements ✅

1. **Performance**: Analysis completes in <10 seconds for simple queries, <30 seconds for complex
2. **Accuracy**: LLM responses are factually correct based on data (measured by manual review)
3. **Test Coverage**: 90%+ code coverage with comprehensive unit and integration tests
4. **Maintainability**: New features can be added without modifying existing code (Open/Closed Principle)
5. **Reliability**: System handles errors gracefully and provides helpful feedback

### Acceptance Tests

Test with all example questions from the spec:
- [ ] "How has Apple's quarterly revenue growth rate evolved over the past five years compared to Microsoft's?"
- [ ] "For each company in my portfolio, what was the year-over-year change in net income?"
- [ ] "Compare the 30-day, 90-day, and 1-year price volatility of Tesla and its top three peers since 2020"
- [ ] (All remaining example questions...)

Each test must:
1. Generate a correct query plan
2. Retrieve appropriate data
3. Provide meaningful insights
4. Synthesize natural language answer
5. Complete within performance targets

---

---

## 12. Future Enhancements

### REST API
```python
# FastAPI endpoints
@app.post("/api/v1/analyze")
async def analyze(request: AnalysisRequest) -> AnalysisResult:
    ...

@app.get("/api/v1/conversations/{id}")
async def get_conversation(id: str) -> Conversation:
    ...
```

### Web UI
- React frontend
- Real-time streaming responses
- Interactive charts
- Saved queries and dashboards

### Advanced Analytics
- Portfolio optimization
- Risk analysis
- Correlation matrices
- Backtesting

### ML Integration
- Price prediction models
- Anomaly detection
- Sentiment analysis from news
- Custom trained models

### Multi-user Support
- User authentication
- Shared portfolios
- Collaborative analysis
- Access control

---

## 13. Appendix

### A. Example LLM Prompts

#### Query Planning Prompt
```
You are a financial data analyst planning how to answer a question.

QUESTION: Compare AAPL and MSFT revenue for the last 5 years

AVAILABLE DATABASE SCHEMA:
- financials.annual: ticker, period_end_date, totalRevenue, netIncome, ...
- company.metadata: ticker, name, sector, industry, currency, country

Your task:
1. Analyze the question complexity
2. Break it into logical steps (queries)
3. Identify dependencies between steps
4. Write SQL for each step
5. Determine if visualization would be helpful

Output JSON plan:
{
  "complexity": "moderate",
  "requires_visualization": true,
  "reasoning": "Need to fetch revenue for both companies, then compare",
  "steps": [
    {
      "step_number": 1,
      "description": "Get AAPL revenue for last 5 years",
      "sql_query": "SELECT ticker, period_end_date, totalRevenue FROM financials.annual WHERE ticker = 'AAPL' AND period_end_date >= DATE_SUB(CURRENT_DATE, INTERVAL 5 YEAR) ORDER BY period_end_date",
      "dependencies": [],
      "expected_columns": ["ticker", "period_end_date", "totalRevenue"]
    },
    {
      "step_number": 2,
      "description": "Get MSFT revenue for last 5 years",
      "sql_query": "SELECT ticker, period_end_date, totalRevenue FROM financials.annual WHERE ticker = 'MSFT' AND period_end_date >= DATE_SUB(CURRENT_DATE, INTERVAL 5 YEAR) ORDER BY period_end_date",
      "dependencies": [],
      "expected_columns": ["ticker", "period_end_date", "totalRevenue"]
    }
  ]
}
```

#### Analysis Prompt
```
You are a financial analyst examining data to answer a question.

ORIGINAL QUESTION: Compare AAPL and MSFT revenue for the last 5 years

DATA RETRIEVED:
Step 1:
- Rows: 5
- Columns: ticker, period_end_date, totalRevenue
- Sample data:
  ticker  period_end_date  totalRevenue
  AAPL    2019-09-30      260174000000
  AAPL    2020-09-30      274515000000
  AAPL    2021-09-30      365817000000
  AAPL    2022-09-30      394328000000
  AAPL    2023-09-30      383285000000

Step 2:
- Rows: 5
- Columns: ticker, period_end_date, totalRevenue
- Sample data:
  ticker  period_end_date  totalRevenue
  MSFT    2019-06-30      125843000000
  MSFT    2020-06-30      143015000000
  MSFT    2021-06-30      168088000000
  MSFT    2022-06-30      198270000000
  MSFT    2023-06-30      211915000000

Analyze the data and extract insights:
{
  "insights": [
    {
      "finding": "Apple's revenue grew from $260B in 2019 to $394B in 2022 (51% growth), but declined to $383B in 2023",
      "significance": "Apple experienced strong growth but recent decline may indicate market saturation or economic headwinds",
      "data_points": ["2019: $260B", "2022 peak: $394B", "2023: $383B"],
      "category": "trend"
    },
    {
      "finding": "Microsoft showed consistent growth from $126B in 2019 to $212B in 2023 (68% growth)",
      "significance": "Microsoft's steady growth suggests diversified revenue streams and strong market position",
      "data_points": ["2019: $126B", "2023: $212B", "68% growth"],
      "category": "trend"
    },
    {
      "finding": "Apple's revenue is approximately 1.8x Microsoft's, but Microsoft is closing the gap",
      "significance": "The revenue gap between companies is narrowing as Microsoft grows faster",
      "data_points": ["2019 ratio: 2.1x", "2023 ratio: 1.8x"],
      "category": "comparison"
    }
  ],
  "confidence": 0.9
}
```

#### Synthesis Prompt
```
You are a financial analyst providing a clear answer to a client's question.

QUESTION: Compare AAPL and MSFT revenue for the last 5 years

INSIGHTS:
1. Apple's revenue grew from $260B in 2019 to $394B in 2022 (51% growth), but declined to $383B in 2023
2. Microsoft showed consistent growth from $126B in 2019 to $212B in 2023 (68% growth)
3. Apple's revenue is approximately 1.8x Microsoft's, but Microsoft is closing the gap

Generate clear answer:

Over the past five years, both Apple and Microsoft have demonstrated strong revenue growth, but with notably different trajectories.

Apple experienced explosive growth from $260B in 2019 to a peak of $394B in 2022 (51% increase), but saw a decline to $383B in 2023. This recent downturn may signal market saturation in their core iPhone business or broader economic pressures.

Microsoft, meanwhile, showed more consistent and sustained growth, increasing revenue from $126B in 2019 to $212B in 2023—a 68% increase over the period. This steady performance suggests strong fundamentals across their diversified portfolio of cloud services, enterprise software, and gaming.

While Apple still leads with approximately 1.8x Microsoft's revenue, this gap has narrowed from 2.1x in 2019. If current trends continue, Microsoft's consistent growth could further close this gap over the next few years.
```

### B. Database Schema Details

All database schemas are documented in Section 3. Key additions for conversation tracking:

**Conversation Tables:**
```sql
-- Conversation tracking
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);

CREATE TABLE conversation_exchanges (
    exchange_id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(conversation_id),
    question TEXT,
    answer TEXT,
    insights JSON,
    query_plan JSON,
    timestamp TIMESTAMP
);

CREATE INDEX idx_conv_exchanges ON conversation_exchanges(conversation_id, timestamp);
```

### C. Performance Benchmarks

**Target Performance:**

| Operation | Target | Notes |
|-----------|--------|-------|
| Simple query (1 step) | <5s | "What is AAPL revenue?" |
| Moderate query (2-3 steps) | <15s | "Compare AAPL vs MSFT" |
| Complex query (4-5 steps) | <30s | "Analyze trends across 10 companies" |
| Visualization generation | <3s | After data retrieval |
| Conversation context loading | <1s | From SQLite |
| Cache hit | <100ms | For repeated queries |
| Data ingestion (50 tickers) | <45s | Concurrent processing |
| Data transformation (1000 tickers) | <60s | Streaming mode |

---

## 14. Conclusion

This specification provides a complete blueprint for FinanGPT. The key architectural principles are:

1. **Agent-based architecture** for multi-step analytical reasoning
2. **LLM sees and analyzes actual data**, not just generates SQL queries
3. **Clean layered architecture** enabling comprehensive testing and maintainability
4. **Structured outputs** at each step for transparency and debugging
5. **Context-aware conversations** for natural, flowing interactions

The system combines powerful data ingestion and transformation capabilities with sophisticated AI-driven analysis to deliver true conversational financial intelligence.

**Next Steps:**
1. Review and approve this specification
2. Set up development environment (Python 3.13+)
3. Begin Phase 1 implementation
4. Iterate and refine based on learnings

---

**Document Version**: 1.0
**Date**: November 12, 2025
**Status**: Ready for Development