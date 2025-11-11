"""Compatibility wrapper for resilience helpers used in tests."""

from src.query_engine.resilience import (
    execute_template,
    get_all_tickers,
    list_templates,
    load_query_templates,
    suggest_tickers,
    validate_ticker,
)


def handle_ollama_failure(error: Exception):
    print("\n⚠️  Ollama is not reachable. Connection error:")
    print(f"   {error}")
    print("\n💡 Fallback options:")
    print("   1. Enter SQL directly (expert mode)")
    print("   2. Use saved query templates")
    print("   3. Exit and fix connection")

    choice = input("\nSelect [1/2/3]: ").strip()

    if choice == "1":
        print("\n📝 Expert Mode: Enter your SQL query directly")
        print("   (Must be a SELECT statement with allowed tables)")
        sql = input("\nSQL> ").strip()
        if sql:
            return sql
        print("No SQL provided.")
        return None

    if choice == "2":
        try:
            templates = load_query_templates()
            print("\n📚 Available query templates:")
            for name, tpl in templates.items():
                print(f"   • {name}: {tpl.get('description', 'No description')}")

            template_name = input("\nTemplate name> ").strip()
            if template_name not in templates:
                print(f"Template '{template_name}' not found.")
                return None

            template = templates[template_name]
            print(f"\n📋 Template: {template.get('description')}")
            print(f"   Required parameters: {', '.join(template.get('params', []))}")

            params = {}
            for param in template.get("params", []):
                default = template.get("defaults", {}).get(param)
                prompt = f"{param}" + (f" (default: {default})" if default else "") + "> "
                value = input(prompt).strip()
                if value:
                    params[param] = value
                elif default is not None:
                    params[param] = default

            final_params = template.get("defaults", {}).copy()
            final_params.update(params)
            sql = template["sql"].format(**final_params)
            print(f"\n📊 Generated SQL: {sql}")
            return sql
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading templates: {exc}")
            return None

    print("Exiting. Please check Ollama connection and try again.")
    return None


def debug_log(message: str, enabled: bool = False) -> None:
    if enabled:
        print(f"[DEBUG] {message}")

__all__ = [
    "execute_template",
    "get_all_tickers",
    "list_templates",
    "load_query_templates",
    "suggest_tickers",
    "validate_ticker",
    "handle_ollama_failure",
    "debug_log",
]
