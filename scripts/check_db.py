import ast
import os

import chromadb


def bruteforce_inspect_db():
    db_path = "data/vector_db"

    if not os.path.exists(db_path):
        print(f"Error: Path {db_path} does not exist.")
        return

    # Connect directly to the raw Chroma persistent client
    client = chromadb.PersistentClient(path=db_path)

    # List all collections to make sure we are hitting the right one
    collections = client.list_collections()
    if not collections:
        print("No collections found in the DB.")
        return

    # LangChain defaults the collection name to 'langchain'
    collection_name = "langchain"
    print(f"--- Accessing Raw Collection: '{collection_name}' ---")
    collection = client.get_collection(name=collection_name)

    # Get everything: documents, metadatas, and ids
    results = collection.get(include=["metadatas", "documents"], limit=30)

    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    if not metadatas:
        print("Raw metadata is empty!")
        return

    print(f"Inspecting {len(metadatas)} raw records...\n")

    for i in range(len(metadatas)):
        meta = metadatas[i]
        doc = documents[i]

        print(f"RECORD #{i}")
        print(f"  Content Snippet: {doc[:70]}...")

        # Print ALL keys in metadata to see if 'predicate' even exists or is misspelled
        print(f"  Available Metadata Keys: {list(meta.keys())}")

        # Bruteforce check for the predicate value
        val = meta.get("predicate", "!! MISSING KEY !!")
        print(f"  [RAW PREDICATE]: '{val}' (Type: {type(val)})")

        # Check roles for stringified dict integrity
        roles_raw = meta.get("srl_roles", "{}")
        try:
            # If this fails, the data was saved in a format ast can't read
            parsed_roles = ast.literal_eval(roles_raw)
            print(f"  [RAW ROLES]: Valid Dict with {len(parsed_roles)} keys")
        except Exception as e:
            print(f"  [RAW ROLES ERROR]: {e} | Raw string: {roles_raw}")

        print("-" * 50)


if __name__ == "__main__":
    bruteforce_inspect_db()
