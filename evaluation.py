from rag_qa import RAGQA

questions = [
    "satellite communication",
    "financial policy",
    "economic growth"
]

for q in questions:
    results = rag.ask(q)

    print("\nQuestion:", q)
    print("Pages:", results["pages"])