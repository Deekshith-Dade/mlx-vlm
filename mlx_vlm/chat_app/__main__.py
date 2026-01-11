import argparse
from .app import ChatApp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Textual Chat App")
    parser.add_argument("--name", default="AI", help="Name of the chat agent")
    args = parser.parse_args()

    app = ChatApp(agent_name=args.name)
    app.run()
