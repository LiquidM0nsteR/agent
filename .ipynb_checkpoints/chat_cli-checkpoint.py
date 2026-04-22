from __future__ import annotations

from backend.rag.config import get_config
from backend.rag.qa_chain import LocalKnowledgeQAChain


def main() -> None:
    config = get_config()
    qa_chain = LocalKnowledgeQAChain.from_config(config)
    print("Local Knowledge QA CLI")
    print("Type `exit` to quit.")

    while True:
        query = input("\nQuery> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain.ask(query)
        print("\nAnswer:")
        print(result["answer"])
        print("\nReferences:")
        for ref in result["references"]:
            location = []
            if ref["page"] is not None:
                location.append(f"page={ref['page']}")
            if ref["section"]:
                location.append(f"section={ref['section']}")
            location_text = ", ".join(location) if location else "location=unknown"
            print(f"- {ref['file_name']} ({location_text}) -> {ref['source_path']}")


if __name__ == "__main__":
    main()
